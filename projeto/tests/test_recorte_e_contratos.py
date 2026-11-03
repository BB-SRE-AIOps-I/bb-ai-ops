"""Verificações locais do caso sintético, sem rede ou chamada a LLM."""

from datetime import datetime, timedelta, timezone
import unittest

from pydantic import ValidationError

from aiops.contratos import Evidencia, Fato, Hipotese, ResumoIncidente
from aiops.dados import (
    carregar_evidencias,
    carregar_metricas,
    selecionar_evidencias,
    selecionar_metricas,
)
from aiops.validacao import validar_evidencias, verificar_resumo


FUSO = timezone(timedelta(hours=-3))


# Cria horários no dia e no fuso do laboratório para tornar as comparações reproduzíveis.
def instante(hora: int, minuto: int = 0) -> datetime:
    return datetime(2026, 11, 3, hora, minuto, tzinfo=FUSO)


# Cria uma observação exclusiva dos testes. mudancas permite variar um campo por caso,
# como retirar o recebimento ou fornecer um horário inválido, sem editar os dados da aula.
def evidencia(**mudancas) -> Evidencia:
    campos = dict(
        id="T01",
        servico="consulta-limites",
        observacao="8 conexões ativas e 35 solicitações em espera.",
        ocorrido_em=instante(10),
        recebido_em=instante(10, 1),
        fonte="fixture exclusiva de teste",
    )
    campos.update(mudancas)
    return Evidencia(**campos)


# Monta uma resposta fixa exclusiva dos testes para investigar as regras de validação.
# Ela não é usada no notebook como substituta de uma resposta real da IA.
def resumo(**mudancas) -> ResumoIncidente:
    campos = dict(
        fatos=[Fato(afirmacao="O pool foi reduzido de 20 para 8.", referencias=["E01"])],
        hipoteses=[
            Hipotese(
                descricao="A redução do pool pode contribuir para espera.",
                referencias=["E01", "E04"],
                limitacao="Associação temporal não confirma causa.",
            )
        ],
        lacunas=["Não há avaliação causal neste recorte."],
        perguntas=["Quais contraprovas o engenheiro verificaria?"],
    )
    campos.update(mudancas)
    return ResumoIncidente(**campos)


# Confere a seleção por horário: só pode entrar o que já estava disponível no recorte.
class CasoETempoTest(unittest.TestCase):
    def setUp(self):
        self.evidencias = carregar_evidencias()

    def test_onze_eventos_tem_ids_unicos_e_data_do_caso(self):
        self.assertEqual({e.id for e in self.evidencias}, {f"E{i:02}" for i in range(1, 12)})
        self.assertEqual(len(self.evidencias), 11)
        for e in self.evidencias:
            self.assertEqual(e.ocorrido_em.date(), instante(10).date())
            self.assertEqual(e.ocorrido_em.utcoffset(), timedelta(hours=-3))
            self.assertEqual(e.recebido_em.utcoffset(), timedelta(hours=-3))
            self.assertGreaterEqual(e.recebido_em, e.ocorrido_em)

    def test_recorte_1004_nao_vaza_recepcao_atrasada_ou_futuro(self):
        selecionadas = selecionar_evidencias(self.evidencias, instante(10, 4))
        self.assertEqual({e.id for e in selecionadas}, {f"E{i:02}" for i in range(1, 9)})

    def test_e09_entra_apenas_ao_ser_recebido(self):
        e09 = next(e for e in self.evidencias if e.id == "E09")
        self.assertEqual(e09.ocorrido_em, instante(10, 1))
        self.assertEqual(e09.recebido_em, instante(10, 6))
        self.assertNotIn("E09", {e.id for e in selecionar_evidencias(self.evidencias, instante(10, 5))})
        self.assertIn("E09", {e.id for e in selecionar_evidencias(self.evidencias, instante(10, 6))})

    def test_mitigacao_e_desfecho_so_entram_nos_horarios_corretos(self):
        antes = {e.id for e in selecionar_evidencias(self.evidencias, instante(10, 11))}
        mitigacao = {e.id for e in selecionar_evidencias(self.evidencias, instante(10, 12))}
        depois = {e.id for e in selecionar_evidencias(self.evidencias, instante(10, 15))}
        self.assertNotIn("E10", antes)
        self.assertNotIn("E11", antes)
        self.assertIn("E10", mitigacao)
        self.assertNotIn("E11", mitigacao)
        self.assertEqual(depois, {e.id for e in self.evidencias})

    def test_exclusao_simula_dado_ausente_sem_alterar_fonte(self):
        original = [e.model_dump() for e in self.evidencias]
        selecionadas = selecionar_evidencias(self.evidencias, instante(10, 6), excluir_ids=("E01", "E04"))
        self.assertNotIn("E01", {e.id for e in selecionadas})
        self.assertNotIn("E04", {e.id for e in selecionadas})
        self.assertEqual([e.model_dump() for e in self.evidencias], original)

    def test_recorte_vazio_e_ausencia_sao_explicitos(self):
        self.assertEqual(selecionar_evidencias(self.evidencias, instante(9, 49)), [])
        self.assertEqual(selecionar_evidencias([], instante(10, 4)), [])

    def test_recebimento_desconhecido_nao_e_presumido(self):
        desconhecida = evidencia(recebido_em=None)
        self.assertEqual(selecionar_evidencias([desconhecida], instante(10, 30)), [])

    def test_instante_sem_fuso_e_rejeitado(self):
        with self.assertRaises((ValueError, ValidationError)):
            selecionar_evidencias(self.evidencias, datetime(2026, 11, 3, 10, 4))

    def test_metricas_so_aparecem_depois_do_fechamento(self):
        metricas = carregar_metricas()
        self.assertEqual(len(metricas), 7)
        for horario, quantidade in [((9, 54), 0), ((9, 55), 1), ((10, 0), 2), ((10, 4), 2), ((10, 5), 3), ((10, 6), 3), ((10, 10), 4), ((10, 25), 7)]:
            with self.subTest(horario=horario):
                self.assertEqual(len(selecionar_metricas(metricas, instante(*horario))), quantidade)
        self.assertEqual(selecionar_metricas([], instante(10, 4)), [])

    def test_metricas_recusam_instante_sem_fuso(self):
        with self.assertRaises((ValueError, ValidationError)):
            selecionar_metricas(carregar_metricas(), datetime(2026, 11, 3, 10, 4))

    def test_numeros_das_metricas_conferem_com_pdf(self):
        metricas = carregar_metricas()
        self.assertEqual([j.inicio for j in metricas], [instante(9, 50), instante(9, 55), instante(10), instante(10, 5), instante(10, 10), instante(10, 15), instante(10, 20)])
        self.assertEqual([j.falhas for j in metricas], [1, 1, 80, 85, 75, 2, 1])
        self.assertEqual([j.p95_ms for j in metricas], [150, 155, 900, 950, 880, 170, 160])
        for janela in metricas:
            self.assertEqual(janela.solicitacoes, 1000)
            self.assertEqual(janela.minutos, 5)


# Confere quais entradas devem ser recusadas antes de produzir qualquer resumo.
class EvidenciaInvalidaTest(unittest.TestCase):
    def test_validacao_preserva_e03_como_observacao_da_fonte(self):
        validadas = validar_evidencias(carregar_evidencias())
        self.assertEqual(len(validadas), 11)
        self.assertIn("E02", {e.id for e in validadas})
        self.assertIn("E03", {e.id for e in validadas})

    def test_entrada_vazia_nao_gera_resumo_de_normalidade(self):
        with self.assertRaises((ValueError, ValidationError)):
            validar_evidencias([])

    def test_id_duplicado_nao_e_confundido_com_evento_duplicado(self):
        with self.assertRaises((ValueError, ValidationError)):
            validar_evidencias([evidencia(), evidencia(observacao="Outra observação")])

    def test_recebimento_antes_do_evento_e_rejeitado(self):
        with self.assertRaises((ValueError, ValidationError)):
            validar_evidencias([evidencia(recebido_em=instante(9, 59))])

    def test_evento_sem_timezone_e_rejeitado(self):
        with self.assertRaises((ValueError, ValidationError)):
            validar_evidencias([evidencia(ocorrido_em=datetime(2026, 11, 3, 10))])

    def test_recebimento_sem_timezone_e_rejeitado(self):
        with self.assertRaises((ValueError, ValidationError)):
            validar_evidencias([evidencia(recebido_em=datetime(2026, 11, 3, 10, 1))])


# Confere campos e referências, incluindo o limite de um ID válido ligado a uma frase falsa.
class ResumoReferenciadoTest(unittest.TestCase):
    def setUp(self):
        self.recorte = selecionar_evidencias(carregar_evidencias(), instante(10, 4))

    def test_resumo_referenciado_passa_sem_conferir_aceite_humano(self):
        verificacao = verificar_resumo(resumo(), self.recorte)
        self.assertTrue(verificacao.valido)
        self.assertEqual(verificacao.problemas, [])
        campos = set(ResumoIncidente.model_fields)
        self.assertEqual(campos, {"fatos", "hipoteses", "lacunas", "perguntas"})

    def test_referencia_inexistente_e_apontada(self):
        resultado = verificar_resumo(resumo(fatos=[Fato(afirmacao="Afirmação", referencias=["E99"])]), self.recorte)
        self.assertFalse(resultado.valido)
        self.assertTrue(resultado.problemas)

    def test_referencia_real_mas_fora_do_recorte_e_apontada(self):
        for referencia in ("E09", "E10", "E11"):
            with self.subTest(referencia=referencia):
                resultado = verificar_resumo(resumo(fatos=[Fato(afirmacao="Afirmação", referencias=[referencia])]), self.recorte)
                self.assertFalse(resultado.valido)

    def test_fato_sem_referencia_e_rejeitado(self):
        try:
            saida = resumo(fatos=[Fato(afirmacao="Afirmação sem fonte", referencias=[])])
        except ValidationError:
            return
        resultado = verificar_resumo(saida, self.recorte)
        self.assertFalse(resultado.valido)

    def test_referencia_de_hipotese_tambem_e_verificada(self):
        hipotese = Hipotese(descricao="Hipótese", referencias=["E99"], limitacao="Não validada")
        resultado = verificar_resumo(resumo(hipoteses=[hipotese]), self.recorte)
        self.assertFalse(resultado.valido)

    def test_dado_ausente_nao_e_referencia_valida(self):
        resultado = verificar_resumo(resumo(), [])
        self.assertFalse(resultado.valido)
        self.assertTrue(resultado.problemas)

    def test_saida_nao_pode_acrescentar_uma_aprovacao(self):
        campos = resumo().model_dump()
        campos["aprovado"] = True
        with self.assertRaises(ValidationError):
            ResumoIncidente.model_validate(campos)

    def test_id_valido_nao_comprova_semantica_da_afirmacao(self):
        # Documenta o limite mecânico: a leitura humana da fonte continua obrigatória.
        saida = resumo(fatos=[Fato(afirmacao="CPU do banco a 95%.", referencias=["E01"])])
        self.assertTrue(verificar_resumo(saida, self.recorte).valido)


if __name__ == "__main__":
    unittest.main()
