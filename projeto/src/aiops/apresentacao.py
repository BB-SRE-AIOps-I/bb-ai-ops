"""Organiza e exibe os dados e a resposta da IA em tabelas no notebook.

As funções recebem informações que já existem: evidências carregadas, resumo
produzido e resultado da verificação. Não leem arquivos, não chamam a IA e não
avaliam a verdade das afirmações. Exibir novamente não repete a inferência.
"""
import pandas as pd
from IPython.display import display, Markdown


# Recebe as evidências já carregadas e organiza seus campos em linhas e colunas.
# Devolve um DataFrame do pandas; o notebook o mostra com display(...).
# A tabela mantém todos os registros recebidos, sem selecionar ou alterar a base.
def tabela_evidencias(evidencias):
    """Retorna uma tabela com ID, serviço, horários e observação de cada evidência."""
    return pd.DataFrame([{
        "ID": e.id, "Serviço": e.servico,
        "Ocorreu": e.ocorrido_em.strftime("%H:%M"),
        "Chegou": e.recebido_em.strftime("%H:%M") if e.recebido_em else "Desconhecido",
        "Observação": e.observacao,
    } for e in evidencias])


# Recebe o resumo que a IA já produziu e a verificação que o programa já realizou.
# Exibe fatos, hipóteses, lacunas e perguntas em tabelas separadas, com um aviso final.
# Esta função apresenta o resultado; a conferência das afirmações exige ler as fontes.
def mostrar_resumo(resumo, verificacao):
    """Mostra as tabelas e os avisos no notebook, sem produzir uma nova análise."""
    display(Markdown("### Rascunho para conferência — revisão humana pendente"))
    display(pd.DataFrame([{"Afirmação": f.afirmacao, "Referências": ", ".join(f.referencias)}
                          for f in resumo.fatos]))
    display(pd.DataFrame([{"Hipótese": h.descricao, "Referências": ", ".join(h.referencias),
                           "Limite": h.limitacao} for h in resumo.hipoteses]))
    display(pd.DataFrame({"Lacunas declaradas": resumo.lacunas}))
    display(pd.DataFrame({"Perguntas de investigação": resumo.perguntas}))
    texto = "Estrutura e IDs conferidos. Leia as fontes: isto não comprova a interpretação." if verificacao.valido else "Problemas: " + "; ".join(verificacao.problemas)
    display(Markdown(texto))
