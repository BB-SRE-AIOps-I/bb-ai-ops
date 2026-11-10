"""Workflow local da Aula 02. Os nós calculam e classificam sinais do caso Pix."""

from datetime import datetime, timedelta
from statistics import median
from typing import Literal, TypedDict

from langgraph.graph import END, START, StateGraph
from pydantic import Field

from .contratos import Contrato, Evidencia, JanelaMetrica
from .dados import carregar_evidencias, carregar_metricas, selecionar_evidencias, selecionar_metricas


Status = Literal["candidato", "sem_desvio", "dados_insuficientes"]
Prioridade = Literal["alta", "revisar"]
Classe = Literal["relacionado", "duplicata", "fora_da_jornada", "contraponto", "fora_do_recorte", "a_verificar"]

# A topologia e a duplicata são informações explícitas do cenário didático.
# Nenhum nó precisa adivinhar dependências ou comparar descrições em linguagem natural.
JORNADA_PIX = ("gateway", "api-transferencias", "consulta-limites", "db-contas")
DUPLICATAS_CONHECIDAS = {"E03": "E02"}


class ParametrosTriagem(Contrato):
    slo_sucesso: float = Field(default=0.999, gt=0, lt=1)
    limiar_taxa_erro: float = Field(default=0.01, gt=0, le=1)
    fator_p95: float = Field(default=2.0, gt=1)
    minutos_associacao_anteriores: int = Field(default=10, ge=0)


class Medicao(Contrato):
    inicio: datetime
    fim: datetime
    solicitacoes: int | None = None
    falhas: int | None = None
    taxa_erro: float | None = None
    burn_rate: float | None = None
    p95_ms: float | None = None
    p95_referencia_ms: float | None = None
    inicios_referencia: tuple[datetime, datetime]
    janelas_referencia: list[JanelaMetrica]
    lacunas: list[str]


class Criterio(Contrato):
    """Registra a comparação individual e o motivo de uma eventual lacuna."""
    indicador: Literal["taxa_erro", "p95"]
    valor: float | None
    limiar: float | None
    resultado: Literal["disparou", "abaixo_do_limiar", "indisponivel"]
    motivo: str


class Classificacao(Contrato):
    evidencia: Evidencia
    classe: Classe
    motivo: str
    original_id: str | None = None


class ItemFila(Contrato):
    jornada: str
    prioridade: Prioridade
    justificativa: str
    referencias: list[str]


class DossieTriagem(Contrato):
    corte: datetime
    status: Status
    medicao: Medicao
    parametros: ParametrosTriagem
    criterios: list[Criterio]
    gatilhos: list[str]
    evidencias_recebidas: list[Evidencia]
    classificacoes: list[Classificacao]
    lacunas: list[str]
    fila: list[ItemFila]


class EstadoTriagem(TypedDict, total=False):
    corte: datetime
    inicio_alvo: datetime
    inicios_referencia: tuple[datetime, datetime]
    parametros: ParametrosTriagem
    evidencias: list[Evidencia]
    janelas: list[JanelaMetrica]
    disponiveis: list[Evidencia]
    fechadas: list[JanelaMetrica]
    medicao: Medicao
    status: Status
    criterios: list[Criterio]
    gatilhos: list[str]
    unicos: list[Evidencia]
    classificacoes: list[Classificacao]
    fila: list[ItemFila]
    dossie: DossieTriagem


def recortar(estado: EstadoTriagem):
    """Enxerga apenas eventos recebidos e janelas encerradas no horário de corte."""
    evidencias = [Evidencia.model_validate(e) for e in estado["evidencias"]]
    if len({e.id for e in evidencias}) != len(evidencias):
        raise ValueError("IDs de evidência repetidos na entrada.")
    inicio = estado["inicio_alvo"]
    referencias = estado["inicios_referencia"]
    if inicio.utcoffset() is None or any(h.utcoffset() is None for h in referencias):
        raise ValueError("A janela-alvo e a referência precisam de fuso horário.")
    if len(set(referencias)) != 2 or any(h + timedelta(minutes=5) > inicio for h in referencias):
        raise ValueError("Informe duas janelas de referência distintas e anteriores à janela-alvo.")
    janelas = [JanelaMetrica.model_validate(j) for j in estado["janelas"]]
    fechadas = selecionar_metricas(janelas, estado["corte"])
    if len({j.inicio for j in fechadas}) != len(fechadas):
        raise ValueError("Há mais de uma janela fechada para o mesmo início.")
    return {
        # Cada consulta começa sua própria classificação, mesmo se receber um estado anterior.
        "unicos": [],
        "classificacoes": [],
        "fila": [],
        "parametros": ParametrosTriagem.model_validate(estado["parametros"]),
        "disponiveis": selecionar_evidencias(evidencias, estado["corte"]),
        "fechadas": fechadas,
    }


def medir(estado: EstadoTriagem):
    """Calcula impacto da janela-alvo e usa duas janelas de cinco minutos como referência."""
    inicio = estado["inicio_alvo"]
    fim = inicio + timedelta(minutes=5)
    alvo = next((j for j in estado["fechadas"] if j.inicio == inicio and j.fim == fim), None)
    # A referência é escolhida à parte da janela-alvo. Ela não avança com o corte.
    esperadas = estado["inicios_referencia"]
    anteriores = sorted(
        (j for j in estado["fechadas"]
         if j.inicio in esperadas and j.minutos == 5
         and j.solicitacoes > 0 and j.p95_ms is not None and j.p95_ms > 0),
        key=lambda j: j.inicio,
    )
    referencia = median(j.p95_ms for j in anteriores) if len(anteriores) == 2 else None
    lacunas = []
    taxa = p95 = None
    if estado["corte"] < fim:
        lacunas.append("A janela-alvo ainda está aberta.")
    elif alvo is None:
        lacunas.append("Falta a medição da janela-alvo.")
    elif alvo.solicitacoes == 0:
        lacunas.append("Janela sem tráfego. Taxa de erro e p95 ficam indefinidos.")
    else:
        taxa = alvo.falhas / alvo.solicitacoes
        p95 = alvo.p95_ms
        if p95 is None:
            lacunas.append("Falta o p95 da janela-alvo.")
    if referencia is None:
        lacunas.append("Faltam duas janelas de referência fechadas com tráfego e p95 positivo.")

    return {"medicao": Medicao(
        inicio=inicio,
        fim=fim,
        solicitacoes=alvo.solicitacoes if alvo else None,
        falhas=alvo.falhas if alvo else None,
        taxa_erro=taxa,
        burn_rate=taxa / (1 - estado["parametros"].slo_sucesso) if taxa is not None else None,
        p95_ms=p95,
        p95_referencia_ms=referencia,
        inicios_referencia=esperadas,
        janelas_referencia=anteriores,
        lacunas=lacunas,
    )}


def comparar(indicador, valor, limiar, motivo_ausencia) -> Criterio:
    """Conserva a diferença entre um valor abaixo do limiar e um teste impossível."""
    if valor is None or limiar is None:
        resultado, motivo = "indisponivel", motivo_ausencia
    elif valor >= limiar:
        resultado, motivo = "disparou", "Valor atingiu ou excedeu o limiar."
    else:
        resultado, motivo = "abaixo_do_limiar", "Valor observado abaixo do limiar."
    return Criterio(indicador=indicador, valor=valor, limiar=limiar, resultado=resultado, motivo=motivo)


def detectar(estado: EstadoTriagem):
    """Um critério disparado basta, mesmo se o outro estiver indisponível."""
    m = estado["medicao"]
    p = estado["parametros"]
    limiar_p95 = p.fator_p95 * m.p95_referencia_ms if m.p95_referencia_ms is not None else None
    criterios = [
        comparar("taxa_erro", m.taxa_erro, p.limiar_taxa_erro,
                 "A taxa exige uma janela fechada com solicitações."),
        comparar("p95", m.p95_ms, limiar_p95,
                 "A comparação exige p95 observado e referência válida."),
    ]
    gatilhos = [c.indicador for c in criterios if c.resultado == "disparou"]
    if gatilhos:
        status = "candidato"
    elif any(c.resultado == "indisponivel" for c in criterios):
        status = "dados_insuficientes"
    else:
        status = "sem_desvio"
    return {"status": status, "criterios": criterios, "gatilhos": gatilhos}


def depois_de_detectar(estado: EstadoTriagem):
    """Só um candidato segue para a formação da fila."""
    return "deduplicar" if estado["status"] == "candidato" else "montar_dossie"


def deduplicar(estado: EstadoTriagem):
    """Guarda o vínculo da cópia conhecida e mantém o original quando recebido."""
    unicos, classificacoes = [], []
    ids = {e.id for e in estado["disponiveis"]}
    for evidencia in estado["disponiveis"]:
        original_id = DUPLICATAS_CONHECIDAS.get(evidencia.id)
        if original_id is None:
            unicos.append(evidencia)
        elif original_id in ids:
            classificacoes.append(Classificacao(
                evidencia=evidencia, classe="duplicata", original_id=original_id,
                motivo=f"Repete o alerta {original_id}. A cópia fica registrada para conferência.",
            ))
        else:
            classificacoes.append(Classificacao(
                evidencia=evidencia, classe="a_verificar", original_id=original_id,
                motivo="O alerta foi marcado como cópia e seu original ainda não está disponível.",
            ))
    return {"unicos": unicos, "classificacoes": classificacoes}


def correlacionar(estado: EstadoTriagem):
    """Aplica a janela de associação e o mapa explícito de dependências do Pix."""
    inicio = estado["inicio_alvo"] - timedelta(minutes=estado["parametros"].minutos_associacao_anteriores)
    classificacoes = list(estado["classificacoes"])
    for evidencia in estado["unicos"]:
        if not inicio <= evidencia.ocorrido_em <= estado["corte"]:
            classe, motivo = "fora_do_recorte", "Ocorrência fora do intervalo de associação."
        elif evidencia.servico in JORNADA_PIX:
            classe, motivo = "relacionado", "Serviço no caminho do Pix e ocorrência dentro do intervalo."
        elif evidencia.servico == "parceiro-externo":
            classe, motivo = "contraponto", "Contexto para verificar a hipótese sobre o parceiro."
        elif evidencia.servico == "cache-marketing":
            classe, motivo = "fora_da_jornada", "O mapa do caso não liga este cache ao caminho do Pix."
        else:
            classe, motivo = "a_verificar", "A relação do serviço com a jornada precisa ser confirmada."
        classificacoes.append(Classificacao(evidencia=evidencia, classe=classe, motivo=motivo))
    return {"classificacoes": sorted(classificacoes, key=lambda c: (c.evidencia.ocorrido_em, c.evidencia.id))}


def priorizar(estado: EstadoTriagem):
    """Traduz os critérios disparados em prioridade para investigação."""
    medicao = estado["medicao"]
    if "taxa_erro" in estado["gatilhos"]:
        prioridade = "alta"
        taxa = f"{medicao.taxa_erro:.2%}".replace(".", ",")
        motivo = f"Falhas na jornada atingiram o limiar de triagem. Taxa observada de {taxa}."
        if "p95" in estado["gatilhos"]:
            motivo += f" O p95 de {medicao.p95_ms:g} ms também atingiu o limiar."
    else:
        prioridade = "revisar"
        motivo = (
            f"O p95 de {medicao.p95_ms:g} ms atingiu o limiar de comparação. "
            "A aula usa uma referência de latência e um SLO de sucesso."
        )
    if medicao.lacunas:
        motivo += " A avaliação contém lacunas registradas no dossiê."
    return {"fila": [ItemFila(
        jornada="Transferência Pix",
        prioridade=prioridade,
        justificativa=motivo,
        referencias=[c.evidencia.id for c in estado["classificacoes"] if c.classe == "relacionado"],
    )]}


def montar_dossie(estado: EstadoTriagem):
    """Devolve o mesmo formato de saída para os três caminhos do grafo."""
    lacunas = list(estado["medicao"].lacunas)
    if estado["status"] == "candidato":
        lacunas.append("A associação por horário e jornada ainda exige investigação de causa.")
        if not estado["fila"][0].referencias:
            lacunas.append("Nenhum evento recebido pôde ser ligado à jornada neste recorte.")
    return {"dossie": DossieTriagem(
        corte=estado["corte"], status=estado["status"], medicao=estado["medicao"],
        parametros=estado["parametros"], criterios=estado["criterios"], gatilhos=estado["gatilhos"],
        evidencias_recebidas=estado["disponiveis"],
        classificacoes=estado.get("classificacoes", []), lacunas=lacunas, fila=estado.get("fila", []),
    )}


def montar_grafo_triagem():
    """Conecta as funções Python e explicita quando há uma fila a produzir."""
    grafo = StateGraph(EstadoTriagem)
    grafo.add_node("recortar", recortar)
    grafo.add_node("medir", medir)
    grafo.add_node("detectar", detectar)
    grafo.add_node("deduplicar", deduplicar)
    grafo.add_node("correlacionar", correlacionar)
    grafo.add_node("priorizar", priorizar)
    grafo.add_node("montar_dossie", montar_dossie)
    grafo.add_edge(START, "recortar")
    grafo.add_edge("recortar", "medir")
    grafo.add_edge("medir", "detectar")
    grafo.add_conditional_edges("detectar", depois_de_detectar, {
        "deduplicar": "deduplicar", "montar_dossie": "montar_dossie",
    })
    grafo.add_edge("deduplicar", "correlacionar")
    grafo.add_edge("correlacionar", "priorizar")
    grafo.add_edge("priorizar", "montar_dossie")
    grafo.add_edge("montar_dossie", END)
    return grafo.compile()


def horario(hora: str) -> datetime:
    """Todos os horários do cenário pertencem ao incidente fictício de 03/11."""
    return datetime.fromisoformat(f"2026-11-03T{hora}:00-03:00")


def entrada_pix(
    corte: datetime,
    *,
    inicio_alvo: datetime = horario("10:00"),
    inicios_referencia: tuple[datetime, datetime] = (horario("09:50"), horario("09:55")),
) -> EstadoTriagem:
    """Prepara uma consulta local com janela, corte e referência independentes."""
    if corte.utcoffset() is None:
        raise ValueError("O corte da triagem precisa de fuso horário.")
    return {
        "corte": corte,
        "inicio_alvo": inicio_alvo,
        "inicios_referencia": inicios_referencia,
        "parametros": ParametrosTriagem(),
        "evidencias": carregar_evidencias(),
        "janelas": carregar_metricas(),
    }
