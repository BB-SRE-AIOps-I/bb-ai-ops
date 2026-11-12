"""Checkpoint local: nova execução tem ID novo; consulta não faz inferência."""
import sqlite3
from uuid import uuid4
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.types import Command
from langsmith import tracing_context
from .dados import LOCAIS
from .workflow import montar_grafo


class SessaoRCA:
    def __init__(self, cliente, indice, *, banco=None, construtor=montar_grafo):
        LOCAIS.mkdir(parents=True, exist_ok=True)
        self.cliente = cliente
        self.conexao = sqlite3.connect(str(banco or LOCAIS / "checkpoints.sqlite"), check_same_thread=False)
        self.saver = SqliteSaver(self.conexao)
        self.grafo = construtor(cliente, indice, self.saver)

    def config(self, ident):
        return {"configurable": {"thread_id": ident}}

    def iniciar(self, entrada):
        ident = f"{self.cliente.provedor}-{uuid4().hex}"
        with tracing_context(enabled=False):
            resultado = self.grafo.invoke(entrada, self.config(ident))
        return ident, resultado

    def consultar(self, ident):
        estado = self.grafo.get_state(self.config(ident))
        if not estado.values:
            raise ValueError("Execução não encontrada neste banco de checkpoints.")
        return estado

    def decidir(self, ident, versao, acao, comentario=""):
        estado = self.consultar(ident)
        if not ident.startswith(self.cliente.provedor + "-"):
            raise ValueError("Retome com o mesmo provedor da execução.")
        chamadas = estado.values.get("chamadas", [])
        if chamadas and chamadas[0]["modelo"] != self.cliente.modelo:
            raise ValueError("Retome com o mesmo modelo da execução.")
        if not any(t.interrupts for t in estado.tasks):
            raise ValueError("Esta execução não tem decisão pendente.")
        if versao != estado.values.get("versao"):
            raise ValueError("A versão mudou. Leia o postmortem atual antes de decidir.")
        if acao not in {"aprovar", "solicitar_revisao"}:
            raise ValueError("Decisão desconhecida.")
        if acao == "solicitar_revisao" and not comentario.strip():
            raise ValueError("Diga o que deve ser revisto.")
        with tracing_context(enabled=False):
            return self.grafo.invoke(Command(resume={"versao": versao, "acao": acao,
                                     "comentario": comentario}), self.config(ident))

    def fechar(self):
        self.conexao.close()
