"""Embeddings locais e Chroma: proximidade documental, não confiança causal."""
import hashlib
import json
import re
from functools import cached_property
from pathlib import Path
import chromadb
from chromadb.config import Settings
from fastembed import TextEmbedding
from .dados import DATA, LOCAIS

MODELO_EMBEDDING = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
CONSULTAS = [
    "Pool de conexões: ocupação, tempo de retenção e fila de espera no Pix",
    "Banco de dados: CPU, latência de consulta e limites de medições pontuais",
    "Parceiro externo: ausência de degradação reportada e evidências necessárias",
]


def carregar_documentos(pasta=DATA / "documentos"):
    """Metadados JSON na primeira linha; seções Markdown conservadas nos trechos."""
    documentos = []
    for caminho in sorted(Path(pasta).glob("*.md")):
        meta, texto = caminho.read_text(encoding="utf-8").split("\n", 1)
        documentos.append({**json.loads(meta), "texto": texto.strip(), "arquivo": caminho.name})
    return documentos


def filtro_jornada():
    return {"$and": [{"vigente": {"$eq": True}}, {"jornada": {"$eq": "pix"}}]}


class IndiceDocumental:
    def __init__(self, pasta=LOCAIS / "chroma"):
        self.pasta = Path(pasta)
        self.cliente = chromadb.PersistentClient(path=str(self.pasta), settings=Settings(anonymized_telemetry=False))
        self.colecao = None

    @cached_property
    def embeddings(self):
        return TextEmbedding(model_name=MODELO_EMBEDDING, cache_dir=str(LOCAIS / "modelos"),
                             threads=4, max_length=128)

    def separar_trechos(self, documentos):
        """Até 100 tokens de conteúdo por trecho; nenhuma truncagem silenciosa."""
        tokenizador = self.embeddings.model.tokenizer
        trechos = []
        for doc in documentos:
            partes = re.split(r"(?m)^## ", doc["texto"])
            for numero, parte in enumerate(partes):
                if not parte.strip():
                    continue
                linhas = parte.strip().splitlines()
                titulo = linhas[0].lstrip("# ")
                palavras = parte.strip().split()
                acumulado = []
                grupos = []
                for palavra in palavras:
                    candidato = " ".join(acumulado + [palavra])
                    if len(tokenizador.encode(candidato).ids) > 100:
                        if not acumulado:
                            raise ValueError("Uma palavra excedeu o limite do embedding.")
                        grupos.append(" ".join(acumulado)); acumulado = [palavra]
                    else:
                        acumulado.append(palavra)
                if acumulado:
                    grupos.append(" ".join(acumulado))
                for bloco, texto in enumerate(grupos):
                    digest = hashlib.sha256(texto.encode()).hexdigest()[:8]
                    ident = f'D{doc["id"]}-v{doc["versao"]}-s{numero}-{bloco}-{digest}'
                    trechos.append({"id": ident, "texto": texto, "metadados": {
                        k: doc[k] for k in ("titulo", "versao", "vigente", "jornada", "servico", "arquivo")
                    } | {"secao": titulo, "tokens": len(tokenizador.encode(texto).ids)}})
        return trechos

    def preparar(self, documentos=None):
        """Cria explicitamente ou reutiliza coleção compatível com manifesto completo."""
        documentos = carregar_documentos() if documentos is None else documentos
        assinatura = hashlib.sha256(json.dumps({"modelo": MODELO_EMBEDDING, "tokens": 100,
             "documentos": documentos}, ensure_ascii=False, sort_keys=True).encode()).hexdigest()
        nome = "rca-" + assinatura[:24]
        try:
            colecao = self.cliente.get_collection(nome, embedding_function=None)
            if colecao.metadata.get("assinatura") != assinatura:
                raise ValueError("Manifesto do índice incompatível.")
        except chromadb.errors.NotFoundError:
            trechos = self.separar_trechos(documentos)
            if not trechos:
                raise ValueError("Corpus vazio; não há índice para preparar.")
            vetores = [v.tolist() for v in self.embeddings.embed([t["texto"] for t in trechos])]
            colecao = self.cliente.create_collection(nome, embedding_function=None,
                    metadata={"assinatura": assinatura, "modelo": MODELO_EMBEDDING, "hnsw:space": "cosine"})
            colecao.add(ids=[t["id"] for t in trechos], documents=[t["texto"] for t in trechos],
                        metadatas=[t["metadados"] for t in trechos], embeddings=vetores)
        if colecao.count() == 0:
            raise ValueError("Índice incompleto. Remova somente a coleção vazia e prepare novamente.")
        self.colecao = colecao
        return {"colecao": nome, "trechos": colecao.count(), "modelo_embedding": MODELO_EMBEDDING,
                "assinatura": assinatura}

    def buscar(self, consultas=CONSULTAS, *, filtro=None):
        """Aplica filtro antes de top-k. Registra ocorrências, duplicatas e seleção."""
        if self.colecao is None:
            raise ValueError("Prepare o índice antes de consultar.")
        if not 1 <= len(consultas) <= 3:
            raise ValueError("Use de uma a três consultas.")
        where = filtro_jornada() if filtro is None else filtro
        vetores = [v.tolist() for v in self.embeddings.embed(consultas)]
        r = self.colecao.query(query_embeddings=vetores, n_results=3, where=where,
                              include=["documents", "metadatas", "distances"])
        ocorrencias, unicos = [], {}
        # Intercala os rankings para que uma consulta não ocupe todo o contexto.
        for rank in range(3):
            for i, consulta in enumerate(consultas):
                if rank >= len(r["ids"][i]):
                    continue
                ident = r["ids"][i][rank]
                ocorrencias.append({"consulta": consulta, "id": ident, "posicao": rank + 1,
                                     "distancia": r["distances"][i][rank]})
                if ident not in unicos:
                    unicos[ident] = {"id": ident, "texto": r["documents"][i][rank],
                                    "metadados": r["metadatas"][i][rank], "consultas": [],
                                    "motivo": "Ordem intercalada dos rankings, com deduplicação por ID."}
                unicos[ident]["consultas"].append(consulta)
        candidatos = list(unicos.values())
        return {"consultas": list(consultas), "ocorrencias": ocorrencias,
                "candidatos": candidatos, "selecionados": candidatos[:6],
                "descartados": [{"id": t["id"], "motivo": "Limite de seis trechos."} for t in candidatos[6:]],
                "duplicatas": len(ocorrencias) - len(candidatos),
                "colecao": self.colecao.name, "filtro": where}
