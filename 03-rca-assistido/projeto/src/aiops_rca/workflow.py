"""Nós pequenos, roteamento explícito e revisão humana separada da inferência."""
import hashlib
import json
from langgraph.graph import StateGraph, START, END
from langgraph.types import interrupt
from .contratos import EstadoRCA, AnaliseRCA, PlanoVerificacoes
from .dados import selecionar_contexto, fontes_observadas


def preparar_contexto(estado):
    contexto = selecionar_contexto(estado["entrada"])
    candidato = bool(estado["entrada"]["dossie"]["fila"])
    return {"contexto": contexto, "status": "investigando" if candidato else "sem_candidato",
            "erro": None, "chamadas": [], "historico_decisoes": [], "comentario_revisao": "",
            "plano_verificacoes": None, "analise": {}, "postmortem": {}, "versao": "",
            "decisao": {}, "linha_tempo": [], "busca": {}, "validacao": {}}


def montar_linha_tempo(estado):
    return {"linha_tempo": [{"id": e["id"], "ocorrido_em": e["ocorrido_em"],
            "recebido_em": e["recebido_em"], "servico": e["servico"], "observacao": e["observacao"]}
            for e in sorted(estado["contexto"]["evidencias"], key=lambda e: (e["ocorrido_em"], e["id"]))]}


def contexto_para_modelo(contexto):
    """Exclusões ficam na investigação local, sem virar fontes oferecidas ao LLM."""
    return {k: v for k, v in contexto.items() if k != "exclusoes"}


def payload_analise(estado):
    """Exatamente o conteúdo recebido pelo modelo; disponível para estudo sem chamar API."""
    return {"contexto": contexto_para_modelo(estado["contexto"]), "documentacao": estado["busca"]["selecionados"],
            "fontes_observadas_permitidas": sorted(fontes_observadas(estado["contexto"])),
            "comentario_revisao": estado.get("comentario_revisao", "")}


def conferir_analise(analise, contexto, documentos):
    """Confere IDs no contexto enviado. Uma referência existente pode ser mal interpretada."""
    problemas = []
    fontes = fontes_observadas(contexto)
    docs = {d["id"] for d in documentos}
    a = AnaliseRCA.model_validate(analise)
    if {h.id for h in a.hipoteses} != {"H1", "H2", "H3"}:
        problemas.append("As três hipóteses devem ter IDs H1, H2 e H3 distintos.")
    for h in a.hipoteses:
        for ref in h.favoraveis + h.contrapontos:
            if ref not in fontes:
                problemas.append(f"{h.id}: observação {ref} não foi enviada.")
        for ref in h.documentos:
            if ref not in docs:
                problemas.append(f"{h.id}: documento {ref} não foi enviado.")
        if not h.limites or not h.falta_verificar:
            problemas.append(f"{h.id}: explicite limites e o que falta verificar.")
    return {"valida": not problemas, "problemas": problemas,
            "limite": "IDs válidos não comprovam interpretação nem causalidade."}


def validar_referencias(estado):
    if estado.get("erro"):
        return {"validacao": {"valida": False, "problemas": ["A inferência falhou."]}}
    return {"validacao": conferir_analise(estado["analise"], estado["contexto"], estado["busca"]["selecionados"])}


def montar_postmortem(estado):
    a = estado["analise"]
    post = {"titulo": "Investigação do incidente Pix fictício", "situacao": "rascunho para revisão",
            "corte_triagem": estado["contexto"]["corte_triagem"],
            "corte_investigacao": estado["contexto"]["corte_investigacao"],
            "impacto": estado["contexto"]["medicao_original"], "linha_tempo": estado["linha_tempo"],
            "hipoteses": a["hipoteses"], "conclusao": a["conclusao"],
            "mitigacao_observada": [e for e in estado["contexto"]["evidencias"] if e["id"] in {"E10", "E11"}],
            "evolucao_metricas": estado["contexto"]["metricas"],
            "limitacoes": a["limitacoes"] + estado["contexto"]["lacunas_triagem"],
            "acoes_propostas": a["acoes_propostas"], "plano_verificacoes": estado.get("plano_verificacoes"),
            "modelo": estado["chamadas"][-1]["modelo"]}
    versao = hashlib.sha256(json.dumps(post, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    return {"postmortem": post, "versao": versao, "status": "aguardando_revisao"}


def revisar_postmortem(estado):
    """Este nó reinicia na retomada; não contém chamada de modelo nem efeito externo."""
    decisao = interrupt({"versao": estado["versao"], "postmortem": estado["postmortem"],
                         "opcoes": ["aprovar", "solicitar_revisao"]})
    if not isinstance(decisao, dict) or decisao.get("versao") != estado["versao"]:
        raise ValueError("A decisão deve identificar a versão exibida do postmortem.")
    acao = decisao.get("acao")
    comentario = str(decisao.get("comentario", "")).strip()
    if acao not in {"aprovar", "solicitar_revisao"}:
        raise ValueError("Escolha aprovar ou solicitar_revisao.")
    if acao == "solicitar_revisao" and not comentario:
        raise ValueError("Informe o que precisa ser revisto.")
    registro = {"acao": acao, "comentario": comentario, "versao": estado["versao"]}
    return {"decisao": registro, "comentario_revisao": comentario,
            "historico_decisoes": estado.get("historico_decisoes", []) + [registro]}


def finalizar(estado):
    return {"status": "revisado", "postmortem": {**estado["postmortem"],
            "situacao": "revisado por decisão humana; hipóteses continuam sujeitas a evidência"}}


def encerrar_sem_analise(estado):
    motivo = "A triagem não produziu candidato." if estado["status"] == "sem_candidato" else "Nenhum documento aplicável foi recuperado."
    return {"status": "sem_analise", "erro": None,
            "validacao": {"valida": False, "problemas": [motivo]}}


def encerrar_com_pendencias(estado):
    return {"status": "falha_modelo" if estado.get("erro") else "referencias_invalidas"}


def gerar_verificacoes(estado, cliente):
    """Serviço de inferência usado pelo nó que o aluno construirá no exercício."""
    ids = fontes_observadas(estado["contexto"]) | {d["id"] for d in estado["busca"]["selecionados"]}
    plano, chamada = cliente.gerar(PlanoVerificacoes,
        {"hipoteses": estado["analise"]["hipoteses"], "fontes_permitidas": sorted(ids),
         "contexto": contexto_para_modelo(estado["contexto"]), "documentacao": estado["busca"]["selecionados"]},
        "Proponha exatamente uma verificação por hipótese H1,H2,H3, em português. "
        "Use uma frase curta por campo, no máximo 220 caracteres. Não repita a narrativa do caso. "
        "Explique dado a coletar, resultado favorável e contrário. Não execute ações, não invente medidas. "
        "Use apenas referencias fornecidas. São propostas para investigação humana.")
    if {v["hipotese_id"] for v in plano["verificacoes"]} != {"H1", "H2", "H3"}:
        raise ValueError("O plano deve cobrir H1, H2 e H3 uma vez cada.")
    if any(ref not in ids for v in plano["verificacoes"] for ref in v["referencias"]):
        raise ValueError("Plano de verificações cita fonte que não foi enviada.")
    return plano, chamada


def configurar_fluxo_rca(cliente, indice, *, tipo_estado=EstadoRCA, recuperar=None):
    """Deixa uma única conexão didática aberta: após validação e antes do postmortem."""
    def recuperar_documentacao(estado):
        busca = recuperar(indice) if recuperar else indice.buscar()
        return {"busca": busca}

    def analisar_hipoteses(estado):
        try:
            analise, chamada = cliente.gerar(AnaliseRCA, payload_analise(estado))
            return {"analise": analise, "erro": None,
                    "chamadas": estado.get("chamadas", []) + [chamada]}
        except Exception as exc:
            # Mensagens de API podem conter informações da conexão: guarde só tipo e orientação.
            return {"erro": {"tipo": type(exc).__name__,
                    "mensagem": "A geração falhou. Confira credenciais, disponibilidade, limites e estrutura da saída."},
                    "analise": {}, "status": "falha_modelo"}

    g = StateGraph(tipo_estado)
    for nome, funcao in [("preparar_contexto", preparar_contexto), ("montar_linha_tempo", montar_linha_tempo),
            ("recuperar_documentacao", recuperar_documentacao), ("analisar_hipoteses", analisar_hipoteses),
            ("validar_referencias", validar_referencias), ("montar_postmortem", montar_postmortem),
            ("revisar_postmortem", revisar_postmortem), ("finalizar", finalizar),
            ("encerrar_sem_analise", encerrar_sem_analise), ("encerrar_com_pendencias", encerrar_com_pendencias)]:
        g.add_node(nome, funcao)
    g.add_edge(START, "preparar_contexto")
    g.add_conditional_edges("preparar_contexto", lambda e: bool(e["entrada"]["dossie"]["fila"]),
                           {True: "montar_linha_tempo", False: "encerrar_sem_analise"})
    g.add_edge("montar_linha_tempo", "recuperar_documentacao")
    g.add_conditional_edges("recuperar_documentacao", lambda e: bool(e["busca"]["selecionados"]),
                           {True: "analisar_hipoteses", False: "encerrar_sem_analise"})
    g.add_edge("analisar_hipoteses", "validar_referencias")
    g.add_edge("montar_postmortem", "revisar_postmortem")
    g.add_conditional_edges("revisar_postmortem", lambda e: e["decisao"]["acao"],
                           {"aprovar": "finalizar", "solicitar_revisao": "analisar_hipoteses"})
    for nome in ("finalizar", "encerrar_sem_analise", "encerrar_com_pendencias"):
        g.add_edge(nome, END)
    return g


def conectar_validacao(grafo, destino):
    grafo.add_conditional_edges("validar_referencias", lambda e: e["validacao"]["valida"],
                               {True: destino, False: "encerrar_com_pendencias"})


def montar_grafo(cliente, indice, checkpointer):
    g = configurar_fluxo_rca(cliente, indice)
    conectar_validacao(g, "montar_postmortem")
    return g.compile(checkpointer=checkpointer)
