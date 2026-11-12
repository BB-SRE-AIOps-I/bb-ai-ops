"""Contratos da resposta do modelo. Estrutura válida não comprova causalidade."""
from typing import Literal, TypedDict, Annotated
from pydantic import BaseModel, ConfigDict, Field


class Contrato(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Hipotese(Contrato):
    id: Literal["H1", "H2", "H3"]
    titulo: str
    mecanismo: str
    favoraveis: list[Annotated[str, Field(pattern=r"^(E[0-9]{2}|M[0-9]{4}|T01)$")]] = Field(description="Somente IDs de observações, por exemplo E01. Nunca frases.")
    contrapontos: list[Annotated[str, Field(pattern=r"^(E[0-9]{2}|M[0-9]{4}|T01)$")]] = Field(description="Somente IDs de observações que limitam a hipótese. Pode ser vazio.")
    documentos: list[str] = Field(description="Somente IDs completos dos trechos D... enviados, sem explicações.")
    limites: list[str]
    falta_verificar: list[str]


class Acao(Contrato):
    proposta: str
    papel_responsavel: str
    prazo_proposto: str
    evidencia_de_conclusao: str


class AnaliseRCA(Contrato):
    hipoteses: list[Hipotese] = Field(min_length=3, max_length=3)
    conclusao: str
    limitacoes: list[str]
    acoes_propostas: list[Acao] = Field(min_length=2, max_length=2)


class Verificacao(Contrato):
    hipotese_id: Literal["H1", "H2", "H3"]
    pergunta: str = Field(max_length=220)
    dado_a_coletar: str = Field(max_length=220)
    resultado_favoravel: str = Field(max_length=220)
    resultado_contrario: str = Field(max_length=220)
    referencias: list[str]


class PlanoVerificacoes(Contrato):
    verificacoes: list[Verificacao] = Field(min_length=3, max_length=3)


class EstadoRCA(TypedDict, total=False):
    entrada: dict
    contexto: dict
    linha_tempo: list[dict]
    busca: dict
    analise: dict
    chamadas: list[dict]
    validacao: dict
    postmortem: dict
    versao: str
    decisao: dict
    historico_decisoes: list[dict]
    comentario_revisao: str
    status: str
    erro: dict | None
