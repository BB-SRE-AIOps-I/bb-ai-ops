"""Lê o caso autocontido e aplica o corte de conhecimento antes da inferência."""
import json
from datetime import datetime, timedelta
from pathlib import Path

PROJETO = Path(__file__).resolve().parents[2]
DATA = PROJETO / "data"
LOCAIS = PROJETO / "dados_locais"


def ler_json(nome):
    return json.loads((DATA / nome).read_text(encoding="utf-8"))


def horario(hora):
    return datetime.fromisoformat(f"2026-11-03T{hora}:00-03:00")


def entrada_pix(corte_triagem="10:06", corte_investigacao="10:25", *, incluir_trace=True):
    """O dossiê não muda; o corte da investigação controla dados adicionais."""
    if corte_triagem not in {"10:02", "10:05", "10:06"}:
        raise ValueError("Escolha um dos três cortes preparados: 10:02, 10:05 ou 10:06.")
    corte = horario(corte_investigacao)
    if corte < horario(corte_triagem):
        raise ValueError("O corte da investigação não pode anteceder a triagem recebida.")
    if incluir_trace and corte < horario("10:25"):
        raise ValueError("T01 é anexo retrospectivo: use 10:25 ou incluir_trace=False.")
    pacote = ler_json(f'triagem-{corte_triagem.replace(":", "")}.json')
    return {"entrada": {**pacote, "corte_investigacao": corte.isoformat(),
                        "incluir_trace": incluir_trace}}


def selecionar_contexto(entrada):
    """Enriquece sem inventar disponibilidade. Exclui cópia e serviço sem relação."""
    corte = datetime.fromisoformat(entrada["corte_investigacao"])
    if corte.utcoffset() is None:
        raise ValueError("O corte precisa de fuso horário.")
    dossie = entrada["dossie"]
    if corte < datetime.fromisoformat(dossie["corte"]):
        raise ValueError("Corte anterior ao dossiê.")
    recebidas = [e for e in ler_json("evidencias.json")
                 if e["recebido_em"] and datetime.fromisoformat(e["recebido_em"]) <= corte]
    # E08, ausente da fila compacta, é um contraponto importante da investigação.
    classes = {c["evidencia"]["id"]: c["classe"] for c in dossie["classificacoes"]}
    evidencias = [{**e, "classe_triagem": classes.get(e["id"], "adicional_retrospectivo")}
                  for e in recebidas if e["id"] not in {"E03", "E05"}]
    exclusoes = [{"id": e["id"], "motivo": "Duplicata conhecida de E02" if e["id"] == "E03"
                  else "Cache de marketing fora da jornada conhecida"}
                 for e in recebidas if e["id"] in {"E03", "E05"}]
    metricas = []
    for m in ler_json("metricas.json"):
        fim = datetime.fromisoformat(m["inicio"]) + timedelta(minutes=m["minutos"])
        if fim <= corte:
            metricas.append({**m, "id": "M" + datetime.fromisoformat(m["inicio"]).strftime("%H%M"),
                            "fim": fim.isoformat(),
                            "taxa_erro": m["falhas"] / m["solicitacoes"] if m["solicitacoes"] else None})
    trace = None
    if entrada["incluir_trace"]:
        if corte < horario("10:25"):
            raise ValueError("Anexo T01 só está liberado para a investigação retrospectiva.")
        trace = ler_json("trace-t01.json")
    return {"corte_triagem": dossie["corte"], "corte_investigacao": corte.isoformat(),
            "medicao_original": dossie["medicao"], "contexto_aula02": entrada["contexto_rca"],
            "evidencias": evidencias, "exclusoes": exclusoes, "metricas": metricas,
            "trace": trace, "lacunas_triagem": dossie["lacunas"]}


def fontes_observadas(contexto):
    ids = {e["id"] for e in contexto["evidencias"]} | {m["id"] for m in contexto["metricas"]}
    if contexto["trace"]:
        ids.add("T01")
    return ids
