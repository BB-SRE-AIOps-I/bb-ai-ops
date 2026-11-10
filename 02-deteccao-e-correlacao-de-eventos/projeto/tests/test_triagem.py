from datetime import datetime

import pytest

from aiops.triagem import DossieTriagem, ParametrosTriagem, entrada_pix, montar_grafo_triagem


def instante(hora):
    return datetime.fromisoformat(f"2026-11-03T{hora}:00-03:00")


@pytest.fixture(scope="module")
def grafo():
    return montar_grafo_triagem()


def executar(grafo, hora):
    return grafo.invoke(entrada_pix(instante(hora)))["dossie"]


def test_janela_aberta_preserva_relato_sem_calcular_taxa(grafo):
    dossie = executar(grafo, "10:02")
    assert dossie.status == "dados_insuficientes"
    assert dossie.medicao.taxa_erro is None and not dossie.fila
    assert {e.id for e in dossie.evidencias_recebidas} == {f"E0{i}" for i in range(1, 8)}


def test_impacto_e_motivos_da_fila(grafo):
    dossie = executar(grafo, "10:05")
    medicao = dossie.medicao
    assert medicao.solicitacoes == 1000 and medicao.falhas == 80
    assert medicao.taxa_erro == pytest.approx(0.08)
    assert medicao.burn_rate == pytest.approx(80)
    assert (medicao.p95_ms, medicao.p95_referencia_ms) == (900, 152.5)
    assert dossie.fila[0].prioridade == "alta"
    assert set(dossie.fila[0].referencias) == {"E01", "E02", "E04", "E06", "E07"}
    classes = {item.evidencia.id: item for item in dossie.classificacoes}
    assert (classes["E03"].classe, classes["E03"].original_id) == ("duplicata", "E02")
    assert classes["E05"].classe == "fora_da_jornada"
    assert classes["E08"].classe == "contraponto"
    assert all(item.motivo for item in classes.values())
    assert DossieTriagem.model_validate_json(dossie.model_dump_json()) == dossie


def test_evidencia_atrasada_e_reconstrucao_posterior(grafo):
    antes, chegada, posterior = (executar(grafo, hora) for hora in ("10:05", "10:06", "10:20"))
    assert antes.medicao == chegada.medicao == posterior.medicao
    assert "E09" not in antes.fila[0].referencias
    assert set(chegada.fila[0].referencias) - set(antes.fila[0].referencias) == {"E09"}
    cpu = next(e for e in chegada.evidencias_recebidas if e.id == "E09")
    assert cpu.ocorrido_em == instante("10:01") and cpu.recebido_em == instante("10:06")
    assert {e.id for e in posterior.evidencias_recebidas} - {e.id for e in chegada.evidencias_recebidas} == {"E10", "E11"}
    estado_posterior = grafo.invoke(entrada_pix(instante("10:20")))
    assert max(j.inicio for j in estado_posterior["fechadas"]) == instante("10:15")


@pytest.mark.parametrize(("falhas", "p95", "status", "gatilhos"), [
    (10, 150, "candidato", ["taxa_erro"]),
    (1, 305, "candidato", ["p95"]),
    (1, 150, "sem_desvio", []),
])
def test_limiares_separados(grafo, falhas, p95, status, gatilhos):
    entrada = entrada_pix(instante("10:05"))
    entrada["janelas"] = [
        j.model_copy(update={"falhas": falhas, "p95_ms": p95}) if j.inicio == entrada["inicio_alvo"] else j
        for j in entrada["janelas"]
    ]
    dossie = grafo.invoke(entrada)["dossie"]
    assert dossie.status == status and dossie.gatilhos == gatilhos
    if gatilhos:
        assert dossie.fila[0].prioridade == ("alta" if "taxa_erro" in gatilhos else "revisar")
    else:
        assert not dossie.fila


@pytest.mark.parametrize("cenario", ["trafego_zero", "sem_alvo"])
def test_dados_insuficientes_nao_viram_normalidade(grafo, cenario):
    entrada = entrada_pix(instante("10:05"))
    if cenario == "trafego_zero":
        entrada["janelas"] = [
            j.model_copy(update={"solicitacoes": 0, "falhas": 0}) if j.inicio == entrada["inicio_alvo"] else j
            for j in entrada["janelas"]
        ]
    else:
        retirar = instante("10:00")
        entrada["janelas"] = [j for j in entrada["janelas"] if j.inicio != retirar]
    dossie = grafo.invoke(entrada)["dossie"]
    assert dossie.status == "dados_insuficientes" and dossie.lacunas and not dossie.fila
    if cenario == "trafego_zero":
        assert dossie.medicao.taxa_erro is None and dossie.medicao.burn_rate is None


def test_relacao_da_duplicata_sem_original_e_topologia_desconhecida(grafo):
    entrada = entrada_pix(instante("10:05"))
    entrada["evidencias"] = [
        e.model_copy(update={"servico": "vizinho-sem-mapa"}) if e.id == "E05" else e
        for e in entrada["evidencias"] if e.id != "E02"
    ]
    dossie = grafo.invoke(entrada)["dossie"]
    por_id = {c.evidencia.id: c for c in dossie.classificacoes}
    assert por_id["E03"].original_id == "E02"
    assert por_id["E03"].classe == por_id["E05"].classe == "a_verificar"
    assert not {"E03", "E05"}.intersection(dossie.fila[0].referencias)


def test_metricas_abrem_candidato_mesmo_sem_eventos(grafo):
    entrada = entrada_pix(instante("10:05"))
    entrada["evidencias"] = []
    dossie = grafo.invoke(entrada)["dossie"]
    assert dossie.status == "candidato" and not dossie.fila[0].referencias
    assert any("Nenhum evento" in lacuna for lacuna in dossie.lacunas)


def test_dois_registros_com_mesmo_id_sao_erro_de_integridade(grafo):
    entrada = entrada_pix(instante("10:05"))
    entrada["evidencias"].append(entrada["evidencias"][0])
    with pytest.raises(ValueError, match="IDs de evidência repetidos"):
        grafo.invoke(entrada)

@pytest.mark.parametrize(("falhas", "status"), [(80, "candidato"), (1, "dados_insuficientes")])
def test_criterios_independentes_sem_baseline(grafo, falhas, status):
    entrada = entrada_pix(instante("10:05"))
    entrada["janelas"] = [
        j.model_copy(update={"falhas": falhas}) if j.inicio == entrada["inicio_alvo"] else j
        for j in entrada["janelas"] if j.inicio != instante("09:55")
    ]
    dossie = grafo.invoke(entrada)["dossie"]
    assert dossie.status == status
    assert dossie.medicao.taxa_erro == pytest.approx(falhas / 1000)
    assert dossie.criterios[1].resultado == "indisponivel"
    assert dossie.medicao.p95_ms == 900
    assert dossie.medicao.p95_referencia_ms is None
    assert bool(dossie.fila) is (falhas == 80)
    if dossie.fila:
        assert dossie.fila[0].prioridade == "alta" and dossie.lacunas


def test_janela_posterior_conserva_referencia(grafo):
    entrada = entrada_pix(instante("10:20"), inicio_alvo=instante("10:15"))
    dossie = grafo.invoke(entrada)["dossie"]
    assert dossie.status == "sem_desvio" and not dossie.fila
    assert dossie.medicao.taxa_erro == pytest.approx(0.002)
    assert dossie.medicao.burn_rate == pytest.approx(2)
    assert dossie.medicao.p95_ms == 170
    assert dossie.medicao.p95_referencia_ms == 152.5
    assert dossie.criterios[1].limiar == 305


def test_contaminacao_da_referencia_muda_so_criterio_de_latencia(grafo):
    corte = instante("10:15")
    alvo = instante("10:10")
    fixo = grafo.invoke(entrada_pix(corte, inicio_alvo=alvo))["dossie"]
    movel = grafo.invoke(entrada_pix(
        corte, inicio_alvo=alvo, inicios_referencia=(instante("10:00"), instante("10:05"))
    ))["dossie"]
    assert fixo.medicao.p95_ms == movel.medicao.p95_ms == 880
    assert movel.medicao.p95_referencia_ms == 925
    assert movel.criterios[1].limiar == 1850
    assert fixo.gatilhos == ["taxa_erro", "p95"] and movel.gatilhos == ["taxa_erro"]


def test_p95_ausente_na_janela_preserva_falhas(grafo):
    entrada = entrada_pix(instante("10:05"))
    entrada["janelas"] = [
        j.model_copy(update={"p95_ms": None}) if j.inicio == entrada["inicio_alvo"] else j
        for j in entrada["janelas"]
    ]
    dossie = grafo.invoke(entrada)["dossie"]
    assert dossie.gatilhos == ["taxa_erro"]
    assert dossie.criterios[1].resultado == "indisponivel"
    assert dossie.fila[0].prioridade == "alta"


def test_referencia_nao_pode_incluir_janela_alvo(grafo):
    entrada = entrada_pix(instante("10:05"), inicios_referencia=(instante("09:55"), instante("10:00")))
    with pytest.raises(ValueError, match="anteriores"):
        grafo.invoke(entrada)

def test_reexecutar_estado_anterior_nao_herda_fila_nem_classificacao(grafo):
    primeiro = grafo.invoke(entrada_pix(instante("10:05")))
    segundo = grafo.invoke({**primeiro, "corte": instante("10:02")})
    assert segundo["dossie"].status == "dados_insuficientes"
    assert not segundo["dossie"].fila and not segundo["dossie"].classificacoes
    assert not segundo["fila"] and not segundo["classificacoes"]


@pytest.mark.parametrize("ajuste", [{"falhas": 1001}, {"p95_ms": float("nan")}])
def test_janela_alterada_precisa_ser_revalidada(grafo, ajuste):
    entrada = entrada_pix(instante("10:05"))
    entrada["janelas"] = [
        j.model_copy(update=ajuste) if j.inicio == entrada["inicio_alvo"] else j
        for j in entrada["janelas"]
    ]
    with pytest.raises(ValueError):
        grafo.invoke(entrada)


def test_fator_p95_precisa_ser_finito():
    with pytest.raises(ValueError):
        ParametrosTriagem(fator_p95=float("inf"))
