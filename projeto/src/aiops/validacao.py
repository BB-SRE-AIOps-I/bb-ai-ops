"""Verificações determinísticas. Não há chamada de modelo neste arquivo."""
from .contratos import Evidencia, ResumoIncidente, Verificacao


# Recebe registros ou objetos e confere se podem entrar no workflow.
# Retorna evidências válidas; entrada vazia ou IDs repetidos geram erro antes da IA.
def validar_evidencias(evidencias) -> list[Evidencia]:
    """Valida estrutura e unicidade de ID, mantendo registros duplicados da fonte.

    E03 tem outro ID e continua na base: sua deduplicação é tema da Aula 02.
    Dois registros com o mesmo ID são um erro de integridade, não um novo evento.
    """
    validadas = [Evidencia.model_validate(e) for e in evidencias]
    ids = [e.id for e in validadas]
    if len(ids) != len(set(ids)):
        raise ValueError("IDs de evidência repetidos na entrada.")
    if not validadas:
        raise ValueError("Nenhuma evidência recebida; não há base para resumir.")
    return validadas


# Recebe a resposta da IA e exatamente as evidências que estavam no seu contexto.
# Confere campos obrigatórios e IDs citados; devolve valido e uma lista de problemas.
# Não compara o sentido de cada frase com a fonte: ID existente não prova a afirmação.
def verificar_resumo(resumo: ResumoIncidente, evidencias) -> Verificacao:
    """Confere estrutura útil e referências contra SOMENTE o contexto enviado.

    Um ID existente pode ser usado em uma frase falsa. Isso exige leitura humana
    da fonte; esta função não promete detectar toda alucinação ou instrução hostil.
    """
    resumo = ResumoIncidente.model_validate(resumo)
    permitidos = {e.id for e in evidencias}
    problemas = []
    if not resumo.fatos:
        problemas.append("O resumo não apresenta fatos para conferir.")
    for numero, fato in enumerate(resumo.fatos, start=1):
        if not fato.afirmacao.strip():
            problemas.append(f"Fato {numero}: afirmação vazia.")
        if not fato.referencias:
            problemas.append(f"Fato {numero}: falta referência.")
        for referencia in fato.referencias:
            if referencia not in permitidos:
                problemas.append(f"Fato {numero}: {referencia} não pertence ao recorte enviado.")
    for numero, hipotese in enumerate(resumo.hipoteses, start=1):
        if not hipotese.descricao.strip() or not hipotese.limitacao.strip():
            problemas.append(f"Hipótese {numero}: descrição ou limitação ausente.")
        if not hipotese.referencias:
            problemas.append(f"Hipótese {numero}: falta referência.")
        for referencia in hipotese.referencias:
            if referencia not in permitidos:
                problemas.append(f"Hipótese {numero}: {referencia} não pertence ao recorte enviado.")
    if not resumo.perguntas or any(not p.strip() for p in resumo.perguntas):
        problemas.append("Informe perguntas de investigação não vazias.")
    return Verificacao(valido=not problemas, problemas=problemas)
