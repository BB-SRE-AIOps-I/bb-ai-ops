"""Contratos dizem que dados aceitamos; não comprovam a verdade de uma frase.

O modelo só preenche ResumoIncidente. Revisão humana e resultado da verificação
pertencem ao programa e nunca são uma autorização produzida pelo modelo.
"""
from datetime import datetime, timedelta
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


# Base comum: o Pydantic confere tipos e campos ao construir os objetos abaixo.
# Recusar campos extras impede, por exemplo, uma saída com um campo de aprovação inventado.
class Contrato(BaseModel):
    model_config = ConfigDict(extra="forbid")


# Define o formato de uma observação: o que ocorreu, em qual serviço e em quais horários.
# Ter a estrutura correta não comprova a veracidade do conteúdo observado.
class Evidencia(Contrato):
    """Uma observação do laboratório, com horários de ocorrência e recebimento."""
    id: str = Field(min_length=1)
    servico: str = Field(min_length=1)
    observacao: str = Field(min_length=1)
    ocorrido_em: datetime
    recebido_em: datetime | None
    fonte: str = Field(min_length=1)

    # O Pydantic chama esta função para conferir os horários informados.
    # Sem fuso, não podemos compará-los corretamente ao horário de corte.
    @field_validator("ocorrido_em", "recebido_em")
    @classmethod
    def exigir_fuso(cls, valor):
        if valor is not None and valor.utcoffset() is None:
            raise ValueError("Informe o fuso horário, por exemplo -03:00.")
        return valor

    # Com os dois campos disponíveis, confere sua ordem: o recebimento não pode
    # ser anterior à ocorrência. Retorna o próprio objeto quando a regra passa.
    @model_validator(mode="after")
    def conferir_recebimento(self):
        if self.recebido_em is not None and self.recebido_em < self.ocorrido_em:
            raise ValueError("Recebimento anterior ao evento: confira os relógios da fonte.")
        return self


# Representa uma contagem agregada em um intervalo, e não uma requisição individual.
# Seus campos permitem conferir quando a janela terminou e se os números são coerentes.
class JanelaMetrica(Contrato):
    """Uma janela completa só fica selecionável depois do seu término."""
    inicio: datetime
    minutos: int = Field(gt=0)
    solicitacoes: int = Field(ge=0)
    falhas: int = Field(ge=0)
    p95_ms: float = Field(ge=0)
    fonte: str

    # Confere automaticamente se o início da janela inclui o fuso horário.
    @field_validator("inicio")
    @classmethod
    def exigir_fuso(cls, valor):
        if valor.utcoffset() is None:
            raise ValueError("A janela precisa de fuso horário.")
        return valor

    # Uma janela não pode registrar mais falhas do que solicitações recebidas.
    @model_validator(mode="after")
    def conferir_contagem(self):
        if self.falhas > self.solicitacoes:
            raise ValueError("Falhas não podem exceder solicitações.")
        return self

    # Calcula o encerramento a partir do início e da duração, sem armazenar outra cópia.
    # @property permite ler janela.fim como um atributo, embora o valor seja calculado.
    @property
    def fim(self):
        return self.inicio + timedelta(minutes=self.minutos)


# Guarda uma afirmação da IA e os IDs citados para conferi-la nas evidências.
# O nome Fato organiza a resposta; ele não certifica que a afirmação seja verdadeira.
class Fato(Contrato):
    """Afirmação do LLM acompanhada dos IDs que devem sustentá-la."""
    afirmacao: str
    referencias: list[str]


# Guarda uma possível explicação, suas referências e o limite do que se pode concluir.
class Hipotese(Contrato):
    """Explicação a investigar, não uma causa declarada como comprovada."""
    descricao: str
    referencias: list[str]
    limitacao: str


# Reúne as quatro partes que pedimos à IA: fatos, hipóteses, lacunas e perguntas.
# Esse contrato organiza a resposta; não concede autoridade operacional ao modelo.
class ResumoIncidente(Contrato):
    """Esta é a única saída pedida ao modelo. Não há campo de execução."""
    fatos: list[Fato]
    hipoteses: list[Hipotese]
    lacunas: list[str]
    perguntas: list[str]


# Guarda o resultado produzido por verificar_resumo: passou nas regras e quais problemas há.
# Essa informação descreve a conferência do programa, sem registrar aceite humano.
class Verificacao(Contrato):
    """Resultado mecânico; valido não significa revisado por uma pessoa."""
    valido: bool
    problemas: list[str]


# Registra qual modelo respondeu, quando, em quanto tempo e quais evidências recebeu.
# O programa preenche esses metadados; a revisão humana começa como pendente.
class Execucao(Contrato):
    """Metadados de uma inferência concluída, sem chave nem conteúdo privado."""
    modelo: str
    executado_em: datetime
    duracao_s: float
    ids_enviados: list[str]
    revisao_humana: str = "PENDENTE"
