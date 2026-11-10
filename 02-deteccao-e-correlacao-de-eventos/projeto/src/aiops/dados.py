"""Lê evidencias.json e metricas.json na pasta data do projeto."""
import json
from datetime import datetime
from pathlib import Path
from .contratos import Evidencia, JanelaMetrica

# Os caminhos partem deste arquivo, não da pasta em que o terminal foi aberto.
PROJETO = Path(__file__).resolve().parents[2]
DATA = PROJETO / "data"


# Lê evidencias.json e transforma cada registro em um objeto Evidencia.
# Retorna a lista completa; selecionar um horário é responsabilidade da próxima função.
def carregar_evidencias() -> list[Evidencia]:
    """Carrega as onze observações e confere seus campos pelo contrato Pydantic."""
    registros = json.loads((DATA / "evidencias.json").read_text(encoding="utf-8"))
    return [Evidencia.model_validate(registro) for registro in registros]


# Recebe a lista, o horário de corte e, opcionalmente, IDs a retirar do experimento.
# Retorna só o que já havia sido recebido até esse horário, em ordem de ocorrência.
# Uma medição que ocorreu antes, mas chegou depois do corte, ainda fica de fora.
def selecionar_evidencias(evidencias, instante: datetime, excluir_ids=()):
    """Seleciona por recebimento; exclusões são variantes explícitas do exercício.

    Não apaga a base original. Recebimento desconhecido não vira disponibilidade
    presumida. A ordenação usa ocorrência para reconstruir a sequência conhecida.
    """
    if instante.utcoffset() is None:
        raise ValueError("O instante da consulta precisa de fuso horário.")
    return sorted([
        e for e in evidencias
        if e.recebido_em is not None and e.recebido_em <= instante and e.id not in excluir_ids
    ], key=lambda e: (e.ocorrido_em, e.id))


# Lê metricas.json: cada registro resume solicitações, falhas e latência em uma janela.
# Devolve objetos JanelaMetrica. As funções retornam dados para os cálculos locais.
def carregar_metricas() -> list[JanelaMetrica]:
    """Carrega as janelas de métricas usadas para discutir tempo e disponibilidade."""
    registros = json.loads((DATA / "metricas.json").read_text(encoding="utf-8"))
    return [JanelaMetrica.model_validate(registro) for registro in registros]


# Recebe as janelas e um horário de corte; devolve somente as que já terminaram.
# Isso evita usar o total de uma janela quando parte dela ainda pertence ao futuro.
def selecionar_metricas(janelas, instante: datetime):
    """Uma janela das 10h às 10h05 não é uma medição completa às 10h02."""
    if instante.utcoffset() is None:
        raise ValueError("O instante da consulta precisa de fuso horário.")
    return [janela for janela in janelas if janela.fim <= instante]
