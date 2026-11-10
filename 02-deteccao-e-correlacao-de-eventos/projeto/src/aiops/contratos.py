"""Formato e consistência dos dados da jornada Pix."""
from datetime import datetime, timedelta
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


# Base comum: o Pydantic confere tipos e campos ao construir os objetos abaixo.
# Recusar campos extras impede, por exemplo, uma saída com um campo de aprovação inventado.
class Contrato(BaseModel):
    model_config = ConfigDict(extra="forbid", revalidate_instances="always", allow_inf_nan=False)


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
    p95_ms: float | None = Field(default=None, ge=0, allow_inf_nan=False)
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


