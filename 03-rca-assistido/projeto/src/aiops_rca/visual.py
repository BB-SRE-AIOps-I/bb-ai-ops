"""Apresentação local de resultados: não chama modelos."""
import html
import pandas as pd
from IPython.display import display, Markdown, HTML


def matriz(estado):
    return pd.DataFrame([{"Hipótese": h["id"] + " · " + h["titulo"], "Mecanismo proposto": h["mecanismo"],
          "A favor": ", ".join(h["favoraveis"]) or "Sem observação direta",
          "Contrapontos": ", ".join(h["contrapontos"]) or "Sem contraponto recebido",
          "Documentos citados": ", ".join(h["documentos"]) or "Nenhum trecho citado",
          "Limites": " / ".join(h["limites"]), "Falta verificar": " / ".join(h["falta_verificar"])}
          for h in estado.get("analise", {}).get("hipoteses", [])])


def mostrar_resultado(estado):
    display(Markdown("**Estado da investigação:** " + estado.get("status", "não iniciado")))
    if estado.get("erro"):
        display(pd.DataFrame([estado["erro"]])); return
    if estado.get("analise"):
        display(matriz(estado))
        # Escape antes de exibir: não executar HTML retornado por um modelo.
        display(HTML('<p style="max-width:760px;line-height:1.6">' + html.escape(estado["analise"]["conclusao"]) + '</p>'))
    if estado.get("validacao"):
        display(pd.DataFrame([estado["validacao"]]))


def tabela_fontes(estado):
    return pd.DataFrame([{"ID": t["id"], "Documento": t["metadados"]["titulo"],
         "Seção": t["metadados"]["secao"], "Trecho enviado": t["texto"]}
         for t in estado["busca"]["selecionados"]])
