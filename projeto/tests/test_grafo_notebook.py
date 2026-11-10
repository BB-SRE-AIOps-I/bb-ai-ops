"""Exercita o grafo canônico do notebook com inferência substituída só neste teste."""

import ast
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import TypedDict
import unittest
from unittest.mock import Mock

from langgraph.graph import StateGraph, START, END

from aiops.contratos import Evidencia, Execucao, Fato, ResumoIncidente, Verificacao
from aiops.dados import carregar_evidencias, selecionar_evidencias
from aiops.validacao import validar_evidencias, verificar_resumo
from test_recorte_e_contratos import instante, resumo


NOTEBOOK = Path(__file__).resolve().parents[2] / "01-fundamentos-aiops" / "01-fundamentos-de-aiops.ipynb"


# Exercita as células reais do grafo, trocando apenas a chamada de IA por um dublê de teste.
# Assim conferimos os caminhos e seus limites sem rede, credenciais ou custo de API.
class GrafoNotebookTest(unittest.TestCase):
    # Antes de cada teste, monta um grafo novo a partir das definições do notebook.
    # Carrega somente as células que definem o estado, os nós e as conexões do grafo.
    def setUp(self):
        self.evidencias = carregar_evidencias()
        self.recorte = selecionar_evidencias(self.evidencias, instante(10, 6))
        execucao = Execucao(
            modelo="dublo-exclusivo-de-teste",
            executado_em=datetime.now(timezone.utc),
            duracao_s=0,
            ids_enviados=[e.id for e in self.recorte],
        )
        self.inferencia = Mock(return_value=(resumo(), execucao))
        namespace = dict(
            TypedDict=TypedDict, datetime=datetime, StateGraph=StateGraph, START=START, END=END,
            Evidencia=Evidencia, ResumoIncidente=ResumoIncidente, Verificacao=Verificacao,
            Execucao=Execucao, validar_evidencias=validar_evidencias,
            selecionar_evidencias=selecionar_evidencias, verificar_resumo=verificar_resumo,
            gerar_resumo=self.inferencia, cliente=object(),
        )
        notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
        nomes = {"Estado", "validar_no", "resumir_no", "depois_da_selecao"}
        selecionadas = []
        for numero, celula in enumerate(notebook["cells"], start=1):
            fonte = "".join(celula["source"])
            if celula["cell_type"] != "code":
                continue
            definicoes = {no.name for no in ast.parse(fonte).body if isinstance(no, (ast.ClassDef, ast.FunctionDef))}
            if definicoes & nomes:
                selecionadas.append(numero)
                exec(compile(fonte, f"{NOTEBOOK.name}:célula{numero}", "exec"), namespace)
        self.assertEqual(len(selecionadas), 4, "As células canônicas do grafo precisam continuar visíveis.")
        self.grafo = namespace["grafo"]

    def test_recorte_normal_termina_pendente_sem_aprovar_ou_executar(self):
        resultado = self.grafo.invoke({"evidencias": self.evidencias, "instante": instante(10, 6)})
        self.assertEqual(resultado["status"], "AGUARDANDO_REVISAO")
        self.assertEqual(resultado["passos"], ["validar", "selecionar", "resumir", "verificar", "apresentar"])
        self.assertEqual(resultado["execucao"].revisao_humana, "PENDENTE")
        self.assertTrue(resultado["verificacao"].valido)
        self.inferencia.assert_called_once()
        enviados = self.inferencia.call_args.args[1]
        self.assertIn("E09", {e.id for e in enviados})
        self.assertFalse({"E10", "E11"} & {e.id for e in enviados})

    def test_recorte_ausente_encerra_sem_inferencia(self):
        resultado = self.grafo.invoke({"evidencias": self.evidencias, "instante": instante(9)})
        self.assertEqual(resultado["status"], "SEM_EVIDENCIAS")
        self.assertEqual(resultado["passos"], ["validar", "selecionar"])
        self.assertNotIn("resumo", resultado)
        self.inferencia.assert_not_called()

    def test_variante_ausente_nao_envia_e09_nem_muda_base(self):
        self.grafo.invoke({"evidencias": self.evidencias, "instante": instante(10, 6), "excluir_ids": ("E09",)})
        enviados = self.inferencia.call_args.args[1]
        self.assertNotIn("E09", {e.id for e in enviados})
        self.assertIn("E09", {e.id for e in self.evidencias})

    def test_referencia_invalida_encerra_sem_apresentar_como_verificada(self):
        invalido = resumo(fatos=[Fato(afirmacao="Afirmação de teste", referencias=["E10"])])
        self.inferencia.return_value = (invalido, self.inferencia.return_value[1])
        resultado = self.grafo.invoke({"evidencias": self.evidencias, "instante": instante(10, 6)})
        self.assertEqual(resultado["status"], "REFERENCIAS_INVALIDAS")
        self.assertFalse(resultado["verificacao"].valido)
        self.assertNotIn("apresentar", resultado["passos"])

    def test_erro_de_modelo_permanece_erro_no_grafo(self):
        erro = TimeoutError("fixture: sem resposta")
        self.inferencia.side_effect = erro
        with self.assertRaises(TimeoutError) as capturada:
            self.grafo.invoke({"evidencias": self.evidencias, "instante": instante(10, 6)})
        self.assertIs(capturada.exception, erro)
        self.inferencia.assert_called_once()

    def test_entrada_invalida_bloqueia_antes_do_modelo(self):
        with self.assertRaises(ValueError):
            self.grafo.invoke({"evidencias": [self.evidencias[0], self.evidencias[0]], "instante": instante(10, 6)})
        self.inferencia.assert_not_called()


if __name__ == "__main__":
    unittest.main()
