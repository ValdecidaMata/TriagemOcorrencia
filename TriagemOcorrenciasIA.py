import pandas as pd
import numpy as np

np.random.seed(42)

tipos = ["Furto", "Roubo", "Ameaça", "Violência Doméstica", "Estelionato", "Tráfico", "Homicídio (tentativa)"]
locais = ["Asa Norte", "Asa Sul", "Ceilândia", "Taguatinga", "Samambaia", "Planaltina", "Sobradinho"]
periodos = ["Madrugada", "Manhã", "Tarde", "Noite"]

def gerar_descricao(tipo):
    base = {
        "Furto": "Relato de subtração sem violência.",
        "Roubo": "Relato de subtração com ameaça/violência.",
        "Ameaça": "Relato de ameaça verbal ou por mensagens.",
        "Violência Doméstica": "Relato de agressão/ameaça em contexto doméstico.",
        "Estelionato": "Relato de golpe por aplicativo, pix ou cartão.",
        "Tráfico": "Relato de possível comercialização de entorpecentes.",
        "Homicídio (tentativa)": "Relato de agressão grave com risco à vida."
    }
    return base.get(tipo, "Relato de ocorrência.")

n = 200
df = pd.DataFrame({
    "tipo": np.random.choice(tipos, n),
    "local": np.random.choice(locais, n),
    "periodo": np.random.choice(periodos, n),
    "tem_arma": np.random.choice([0,1], n, p=[0.85, 0.15]),
    "vitima_ferida": np.random.choice([0,1], n, p=[0.75, 0.25]),
    "historico_reincidencia": np.random.choice([0,1], n, p=[0.7, 0.3])
})

df["descricao"] = df["tipo"].apply(gerar_descricao)

# Regra didática para gerar "prioridade" (alvo)
def definir_prioridade(row):
    score = 0
    if row["tipo"] in ["Homicídio (tentativa)", "Tráfico", "Violência Doméstica", "Roubo"]:
        score += 2
    if row["tem_arma"] == 1:
        score += 2
    if row["vitima_ferida"] == 1:
        score += 2
    if row["historico_reincidencia"] == 1:
        score += 1
    
    if score >= 5:
        return "Alta"
    elif score >= 3:
        return "Média"
    else:
        return "Baixa"

df["prioridade"] = df.apply(definir_prioridade, axis=1)

df.head(10)


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier

X = df[["tipo","local","periodo","tem_arma","vitima_ferida","historico_reincidencia"]]
y = df["prioridade"]

# colunas categóricas e numéricas
cat_cols = ["tipo","local","periodo"]
num_cols = ["tem_arma","vitima_ferida","historico_reincidencia"]

preprocess = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ("num", "passthrough", num_cols)
    ]
)

model = RandomForestClassifier(n_estimators=200, random_state=42)

pipe = Pipeline(steps=[
    ("prep", preprocess),
    ("model", model)
])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

pipe.fit(X_train, y_train)
pred = pipe.predict(X_test)

print(classification_report(y_test, pred))
print("Matriz de confusão:\n", confusion_matrix(y_test, pred))

nova = pd.DataFrame([{
    "tipo": "Violência Doméstica",
    "local": "Ceilândia",
    "periodo": "Noite",
    "tem_arma": 1,
    "vitima_ferida": 0,
    "historico_reincidencia": 1
}])

pipe.predict(nova)[0]


kb = [
    {
        "tema": "Preservação de Local",
        "conteudo": "Em ocorrências com risco à vida ou crime grave, orientar a preservação do local e acionar equipe competente. Evitar contaminação de vestígios."
    },
    {
        "tema": "Violência Doméstica",
        "conteudo": "Priorizar a segurança da vítima, avaliar risco imediato, orientar registro e medidas protetivas conforme protocolos vigentes."
    },
    {
        "tema": "Estelionato",
        "conteudo": "Coletar evidências digitais (comprovantes, prints, contas), orientar preservação de registros e canais formais para bloqueio/contestação quando aplicável."
    },
    {
        "tema": "Ameaça",
        "conteudo": "Registrar circunstâncias, identificar meio (presencial/mensagem), avaliar risco e orientar preservação de evidências (mensagens, áudios)."
    }
]

def recuperar_contexto(pergunta, kb):
    p = pergunta.lower()
    hits = []
    for item in kb:
        if item["tema"].lower() in p:
            hits.append(item["conteudo"])
    if not hits:
        # fallback simples: pega 1 conteúdo “mais geral”
        hits = [kb[0]["conteudo"]]
    return "\n".join(hits)

pergunta = "Como agir em um caso de Violência Doméstica?"
contexto = recuperar_contexto(pergunta, kb)
contexto

prompt = f"""
Você é um assistente de triagem e orientação.
Use APENAS o contexto abaixo para responder, sem inventar.
Se não houver informação, diga: "Não consta na base".
CONTEXTO:
{contexto}

PERGUNTA:
{pergunta}

RESPOSTA:
"""
print(prompt)


def triagem_completa(ocorrencia_dict, pergunta_orientacao):
    # 1) prioridade via ML
    o = pd.DataFrame([ocorrencia_dict])
    prioridade = pipe.predict(o)[0]
    
    # 2) contexto via RAG simples
    contexto = recuperar_contexto(pergunta_orientacao, kb)
    
    return {
        "prioridade_prevista": prioridade,
        "contexto_recuperado": contexto
    }

resultado = triagem_completa(
    {
        "tipo": "Estelionato",
        "local": "Asa Sul",
        "periodo": "Manhã",
        "tem_arma": 0,
        "vitima_ferida": 0,
        "historico_reincidencia": 0
    },
    "Quais evidências coletar em Estelionato?"
)

resultado
