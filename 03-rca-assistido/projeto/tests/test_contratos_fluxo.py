"""Invariantes do caso e da revisão persistida, sem inferência nem downloads."""
from copy import deepcopy
import json
from unittest.mock import Mock

import chromadb
from chromadb.config import Settings
from langsmith import tracing_context
import numpy as np
from pydantic import ValidationError
import pytest

from aiops_rca.contratos import AnaliseRCA, PlanoVerificacoes
from aiops_rca.dados import entrada_pix, selecionar_contexto
from aiops_rca.rag import IndiceDocumental
from aiops_rca.sessao import SessaoRCA
from aiops_rca.workflow import conferir_analise, gerar_verificacoes, payload_analise


DOCUMENTO_ENVIADO = "D-trecho-de-teste"


@pytest.fixture
def analise():
    acao = {
        "proposta": "Medir o tempo de espera",
        "papel_responsavel": "SRE",
        "prazo_proposto": "Próximo plantão",
        "evidencia_de_conclusao": "Amostras com horário e origem",
    }
    return {
        "hipoteses": [
            {
                "id": ident,
                "titulo": titulo,
                "mecanismo": "Mecanismo possível",
                "favoraveis": ["E01"],
                "contrapontos": [],
                "documentos": [DOCUMENTO_ENVIADO],
                "limites": ["Cobertura parcial"],
                "falta_verificar": ["Obter amostras adicionais"],
            }
            for ident, titulo in (("H1", "Pool"), ("H2", "Banco"), ("H3", "Parceiro"))
        ],
        "conclusao": "Hipóteses em investigação.",
        "limitacoes": ["Cobertura parcial"],
        "acoes_propostas": [acao, {**acao, "proposta": "Comparar os resultados com a referência"}],
    }


class ClienteDeTeste:
    provedor = "teste"
    modelo = "fixture"

    def __init__(self, analise):
        self.analise = analise
        self.pedidos = []

    def gerar(self, contrato, dados):
        self.pedidos.append(deepcopy(dados))
        resposta = deepcopy(self.analise)
        if dados["comentario_revisao"]:
            resposta["conclusao"] = "Análise revista. As hipóteses seguem em investigação."
        return contrato.model_validate(resposta).model_dump(), {"modelo": self.modelo}


class IndiceDeTeste:
    def __init__(self):
        self.consultas = 0

    def buscar(self):
        self.consultas += 1
        return {"selecionados": [{"id": DOCUMENTO_ENVIADO}]}


def contexto_no_corte(corte, *, incluir_trace=False):
    return selecionar_contexto(entrada_pix("10:05", corte, incluir_trace=incluir_trace)["entrada"])


def test_corte_atualiza_conhecimento_sem_mover_a_janela():
    antes = contexto_no_corte("10:05")
    depois = contexto_no_corte("10:06")
    ids_antes = {item["id"] for item in antes["evidencias"]}
    ids_depois = {item["id"] for item in depois["evidencias"]}

    assert ids_antes == {"E01", "E02", "E04", "E06", "E07", "E08"}
    assert ids_depois == ids_antes | {"E09"}
    assert antes["medicao_original"] == depois["medicao_original"]
    assert depois["medicao_original"]["taxa_erro"] == 0.08
    assert antes["trace"] is depois["trace"] is None
    assert {item["id"] for item in depois["metricas"]} == {"M0950", "M0955", "M1000"}


def test_recuperacao_fecha_em_outro_horario_e_trace_e_retrospectivo():
    antes = contexto_no_corte("10:19")
    depois = contexto_no_corte("10:20")
    ids_antes = {item["id"] for item in antes["metricas"]}
    ids_depois = {item["id"] for item in depois["metricas"]}

    assert ids_depois - ids_antes == {"M1015"}
    assert antes["medicao_original"] == depois["medicao_original"]
    assert {item["id"] for item in depois["evidencias"]} >= {"E10", "E11"}
    with pytest.raises(ValueError):
        contexto_no_corte("10:20", incluir_trace=True)
    trace = contexto_no_corte("10:25", incluir_trace=True)["trace"]
    assert trace["id"] == "T01"
    assert trace["recebido_em"] is None
    assert trace["resultado_requisicao"] is None
    assert {span["id"]: span["pai"] for span in trace["spans"]}["espera-conexao"] == "consulta-limites"


@pytest.mark.parametrize(
    ("campo", "fonte"),
    [
        ("favoraveis", "E03"),
        ("favoraveis", "E09"),
        ("favoraveis", "T01"),
        ("documentos", "E01"),
        ("documentos", "D-recuperado-mas-fora-do-payload"),
    ],
)
def test_citacao_precisa_estar_no_conjunto_enviado_para_o_campo(analise, campo, fonte):
    contexto = contexto_no_corte("10:05")
    documentos = [{"id": DOCUMENTO_ENVIADO}]
    assert conferir_analise(analise, contexto, documentos)["valida"]

    analise["hipoteses"][0][campo].append(fonte)
    validacao = conferir_analise(analise, contexto, documentos)
    assert not validacao["valida"]
    assert any(fonte in problema for problema in validacao["problemas"])


def test_contrato_rejeita_documento_no_campo_de_observacoes(analise):
    analise["hipoteses"][0]["favoraveis"].append(DOCUMENTO_ENVIADO)
    with pytest.raises(ValidationError) as erro:
        AnaliseRCA.model_validate(analise)
    assert erro.value.errors()[0]["loc"] == ("hipoteses", 0, "favoraveis", 1)


def test_analise_e_plano_preservam_exclusoes_apenas_no_estado(analise):
    contexto = selecionar_contexto(entrada_pix()["entrada"])
    estado = {"contexto": contexto, "analise": analise,
              "busca": {"selecionados": [{"id": DOCUMENTO_ENVIADO}]}}
    original = deepcopy(estado)
    cliente = Mock()
    plano = PlanoVerificacoes(verificacoes=[
        {"hipotese_id": hipotese["id"], "pergunta": "Houve espera?",
         "dado_a_coletar": "Tempos de aquisição", "resultado_favoravel": "Espera maior",
         "resultado_contrario": "Espera estável", "referencias": ["E04"]}
        for hipotese in analise["hipoteses"]
    ]).model_dump()
    cliente.gerar.return_value = (plano, {"modelo": "fixture"})

    pedido_analise = payload_analise(estado)
    gerar_verificacoes(estado, cliente)
    contrato, pedido_plano, _ = cliente.gerar.call_args.args

    assert contrato is PlanoVerificacoes
    excluidos = {e["id"] for e in contexto["exclusoes"]}
    assert excluidos == {"E03", "E05"}
    assert estado == original
    contexto_esperado = {k: v for k, v in contexto.items() if k != "exclusoes"}
    for pedido, campo in ((pedido_analise, "fontes_observadas_permitidas"),
                          (pedido_plano, "fontes_permitidas")):
        assert pedido["contexto"] == contexto_esperado
        assert excluidos.isdisjoint(pedido[campo])
        assert {"E04", "E09", "T01"} <= set(pedido[campo])
        assert all(ident not in json.dumps(pedido) for ident in excluidos)


def test_chroma_filtra_antes_de_escolher_os_tres_mais_proximos():
    class EmbeddingDeTeste:
        def embed(self, consultas):
            return [np.array([1.0, 0.0]) for _ in consultas]

    cliente = chromadb.EphemeralClient(settings=Settings(anonymized_telemetry=False))
    colecao = cliente.create_collection("corte-de-teste", embedding_function=None)
    colecao.add(
        ids=["marketing", "pix-obsoleto", "pix-1", "pix-2", "pix-3"],
        documents=["Trecho de teste"] * 5,
        metadatas=[
            {"jornada": "marketing", "vigente": True},
            {"jornada": "pix", "vigente": False},
            *({"jornada": "pix", "vigente": True} for _ in range(3)),
        ],
        embeddings=[[1.0, 0.0], [1.0, 0.1], [1.0, 0.2], [1.0, 0.3], [1.0, 0.4]],
    )
    indice = object.__new__(IndiceDocumental)
    indice.colecao = colecao
    indice.embeddings = EmbeddingDeTeste()

    sem_restricao = indice.buscar(["pool"], filtro={"jornada": {"$in": ["pix", "marketing"]}})
    filtrada = indice.buscar(["pool"])

    assert {t["id"] for t in sem_restricao["selecionados"]} == {"marketing", "pix-obsoleto", "pix-1"}
    assert {t["id"] for t in filtrada["selecionados"]} == {"pix-1", "pix-2", "pix-3"}


def test_corte_com_janela_aberta_isola_execucao_e_nao_busca_nem_gera(tmp_path, analise):
    cliente = ClienteDeTeste(analise)
    indice = IndiceDeTeste()
    sessao = SessaoRCA(cliente, indice, banco=tmp_path / "checkpoints.sqlite")
    try:
        ident, resultado = sessao.iniciar(entrada_pix())
        outro_ident, sem_analise = sessao.iniciar(entrada_pix("10:02", "10:02", incluir_trace=False))

        assert outro_ident != ident
        assert resultado["status"] == "aguardando_revisao"
        assert sem_analise["status"] == "sem_analise"
        assert not sem_analise.get("postmortem")
        assert not sem_analise.get("decisao")
        assert not sem_analise["chamadas"]
        assert len(cliente.pedidos) == indice.consultas == 1

        sessao.decidir(ident, resultado["versao"], "aprovar")
        with tracing_context(enabled=False):
            no_mesmo_thread = sessao.grafo.invoke(
                entrada_pix("10:02", "10:02", incluir_trace=False), sessao.config(ident)
            )
        assert no_mesmo_thread["status"] == "sem_analise"
        campos_derivados = ("postmortem", "versao", "analise", "decisao", "linha_tempo", "busca")
        assert all(not no_mesmo_thread.get(campo) for campo in campos_derivados)
        assert len(cliente.pedidos) == indice.consultas == 1
    finally:
        sessao.fechar()


def test_checkpoint_reabre_valida_versao_e_revisa_sem_repetir_aprovacao(tmp_path, analise):
    banco = tmp_path / "checkpoints.sqlite"
    cliente = ClienteDeTeste(analise)
    sessao = SessaoRCA(cliente, IndiceDeTeste(), banco=banco)
    try:
        ident, rascunho = sessao.iniciar(entrada_pix())
        assert rascunho["status"] == "aguardando_revisao"
        assert len(cliente.pedidos) == 1
    finally:
        sessao.fechar()

    outro_cliente = ClienteDeTeste(analise)
    outro_indice = IndiceDeTeste()
    sessao = SessaoRCA(outro_cliente, outro_indice, banco=banco)
    try:
        versao = sessao.consultar(ident).values["versao"]
        assert versao == rascunho["versao"]
        with pytest.raises(ValueError):
            sessao.decidir(ident, "versao-errada", "aprovar")
        with pytest.raises(ValueError):
            sessao.decidir(ident, versao, "solicitar_revisao", " ")
        assert not outro_cliente.pedidos

        comentario = "Explicite a cobertura parcial das amostras."
        revisao = sessao.decidir(ident, versao, "solicitar_revisao", comentario)
        assert revisao["status"] == "aguardando_revisao"
        assert revisao["versao"] != versao
        assert outro_cliente.pedidos[0]["comentario_revisao"] == comentario
        with pytest.raises(ValueError):
            sessao.decidir(ident, versao, "aprovar")

        final = sessao.decidir(ident, revisao["versao"], "aprovar")
        assert final["status"] == "revisado"
        assert [item["acao"] for item in final["historico_decisoes"]] == ["solicitar_revisao", "aprovar"]
        assert len(final["chamadas"]) == 2
        assert len(outro_cliente.pedidos) == 1
        assert outro_indice.consultas == 0
        with pytest.raises(ValueError):
            sessao.decidir(ident, revisao["versao"], "aprovar")
    finally:
        sessao.fechar()
