"""Dublês usados só nos testes; nenhum teste faz inferência ou lê credencial real."""

import json
import unittest
from unittest.mock import Mock, patch

from pydantic import ValidationError

from aiops import modelo
from aiops.contratos import ResumoIncidente
from aiops.dados import PROJETO, carregar_evidencias, selecionar_evidencias
from test_recorte_e_contratos import instante, resumo


# Verifica configuração, montagem das mensagens e tratamento de falhas da integração.
# Mock e patch substituem objetos externos só nestes testes; nenhuma API é chamada.
class ModeloSemRedeTest(unittest.TestCase):
    # Cada teste recebe o mesmo recorte temporal, carregado novamente para evitar interferência.
    def setUp(self):
        self.recorte = selecionar_evidencias(carregar_evidencias(), instante(10, 4))

    def test_sem_chave_bloqueia_antes_de_criar_cliente(self):
        with patch.object(modelo, "dotenv_values", return_value={}) as ler, patch.object(modelo, "ChatOpenAI") as chat:
            with self.assertRaises(ValueError):
                modelo.criar_cliente()
        ler.assert_called_once_with(PROJETO / ".env")
        chat.assert_not_called()

    def test_cliente_recebe_configuracao_sem_retries_ocultos(self):
        config = {"OPENAI_API_KEY": "fixture-sem-validade", "OPENAI_MODEL": "modelo-de-teste", "MODEL_TIMEOUT": "42"}
        with patch.object(modelo, "dotenv_values", return_value=config), patch.object(modelo, "ChatOpenAI") as chat:
            modelo.criar_cliente()
        argumentos = chat.call_args.kwargs
        self.assertEqual(argumentos["model"], "modelo-de-teste")
        self.assertEqual(argumentos["timeout"], 42)
        self.assertEqual(argumentos["max_retries"], 0)
        self.assertTrue(argumentos["use_responses_api"])

    def test_timeout_fora_da_faixa_e_rejeitado_sem_cliente(self):
        for timeout in ("0", "181", "invalido"):
            with self.subTest(timeout=timeout):
                config = {"OPENAI_API_KEY": "fixture-sem-validade", "MODEL_TIMEOUT": timeout}
                with patch.object(modelo, "dotenv_values", return_value=config), patch.object(modelo, "ChatOpenAI") as chat:
                    with self.assertRaises(ValueError):
                        modelo.criar_cliente()
                chat.assert_not_called()

    def test_contexto_e_apenas_recorte_json_sem_metricas_futuras(self):
        mensagens = modelo.mensagens_resumo(self.recorte)
        self.assertEqual([papel for papel, _ in mensagens], ["system", "human"])
        contexto = json.loads(mensagens[1][1])
        self.assertEqual(contexto, [e.model_dump(mode="json") for e in self.recorte])
        self.assertEqual({e["id"] for e in contexto}, {f"E{i:02}" for i in range(1, 9)})

    def test_resposta_tipificada_preserva_pendencia_e_proveniencia(self):
        cliente = Mock(model_name="modelo-de-teste")
        cliente.with_structured_output.return_value.invoke.return_value = resumo()
        resposta, execucao = modelo.gerar_resumo(cliente, self.recorte)
        self.assertIsInstance(resposta, ResumoIncidente)
        self.assertEqual(execucao.modelo, "modelo-de-teste")
        self.assertEqual(execucao.ids_enviados, [e.id for e in self.recorte])
        self.assertEqual(execucao.revisao_humana, "PENDENTE")
        self.assertGreaterEqual(execucao.duracao_s, 0)
        self.assertIsNotNone(execucao.executado_em.utcoffset())
        cliente.with_structured_output.assert_called_once_with(ResumoIncidente, method="json_schema")
        cliente.with_structured_output.return_value.invoke.assert_called_once_with(modelo.mensagens_resumo(self.recorte))

    def test_falha_modelo_e_propagada_sem_resposta_alternativa(self):
        cliente = Mock(model_name="modelo-de-teste")
        falha = TimeoutError("fixture: indisponibilidade do provedor")
        cliente.with_structured_output.return_value.invoke.side_effect = falha
        with self.assertRaises(TimeoutError) as capturada:
            modelo.gerar_resumo(cliente, self.recorte)
        self.assertIs(capturada.exception, falha)
        cliente.with_structured_output.return_value.invoke.assert_called_once()

    def test_saida_estruturada_invalida_e_propagada(self):
        cliente = Mock(model_name="modelo-de-teste")
        cliente.with_structured_output.return_value.invoke.return_value = {"mensagem": "tudo normal"}
        with self.assertRaises(ValidationError):
            modelo.gerar_resumo(cliente, self.recorte)
        cliente.with_structured_output.return_value.invoke.assert_called_once()


if __name__ == "__main__":
    unittest.main()
