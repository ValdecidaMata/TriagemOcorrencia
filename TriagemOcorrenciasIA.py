{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "provenance": [],
      "authorship_tag": "ABX9TyNLLnUo3MQ26/iqk1raMBro",
      "include_colab_link": true
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    },
    "language_info": {
      "name": "python"
    }
  },
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "view-in-github",
        "colab_type": "text"
      },
      "source": [
        "<a href=\"https://colab.research.google.com/github/ValdecidaMata/TriagemOcorrencia/blob/main/TriagemOcorrenciasIA.ipynb\" target=\"_parent\"><img src=\"https://colab.research.google.com/assets/colab-badge.svg\" alt=\"Open In Colab\"/></a>"
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "1\tUm classificador de prioridade (Baixa/Média/Alta) com Machine Learning\n",
        "2\tUm mini-RAG que responde perguntas usando uma “base de conhecimento” (texto) de procedimentos (fictícios/didáticos)\n"
      ],
      "metadata": {
        "id": "-YFheWOtwY0u"
      }
    },
    {
      "cell_type": "code",
      "execution_count": 94,
      "metadata": {
        "id": "Otq5cqyBvWEU"
      },
      "outputs": [],
      "source": [
        "import pandas as pd\n",
        "import numpy as np"
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "np.random.seed(42)\n",
        "\n",
        "tipos = [\"Furto\", \"Roubo\", \"Ameaça\", \"Violência Doméstica\", \"Estelionato\", \"Tráfico\", \"Homicídio (tentativa)\"]\n",
        "locais = [\"Asa Norte\", \"Asa Sul\", \"Ceilândia\", \"Taguatinga\", \"Samambaia\", \"Planaltina\", \"Sobradinho\"]\n",
        "periodos = [\"Madrugada\", \"Manhã\", \"Tarde\", \"Noite\"]\n",
        "\n",
        "def gerar_descricao(tipo):\n",
        "    base = {\n",
        "        \"Furto\": \"Relato de subtração sem violência.\",\n",
        "        \"Roubo\": \"Relato de subtração com ameaça/violência.\",\n",
        "        \"Ameaça\": \"Relato de ameaça verbal ou por mensagens.\",\n",
        "        \"Violência Doméstica\": \"Relato de agressão/ameaça em contexto doméstico.\",\n",
        "        \"Estelionato\": \"Relato de golpe por aplicativo, pix ou cartão.\",\n",
        "        \"Tráfico\": \"Relato de possível comercialização de entorpecentes.\",\n",
        "        \"Homicídio (tentativa)\": \"Relato de agressão grave com risco à vida.\"\n",
        "    }\n",
        "    return base.get(tipo, \"Relato de ocorrência.\")\n",
        "\n",
        "n = 200\n",
        "df = pd.DataFrame({\n",
        "    \"tipo\": np.random.choice(tipos, n),\n",
        "    \"local\": np.random.choice(locais, n),\n",
        "    \"periodo\": np.random.choice(periodos, n),\n",
        "    \"tem_arma\": np.random.choice([0,1], n, p=[0.85, 0.15]),\n",
        "    \"vitima_ferida\": np.random.choice([0,1], n, p=[0.75, 0.25]),\n",
        "    \"historico_reincidencia\": np.random.choice([0,1], n, p=[0.7, 0.3])\n",
        "})\n",
        "\n",
        "df[\"descricao\"] = df[\"tipo\"].apply(gerar_descricao)\n",
        "\n",
        "# Regra didática para gerar \"prioridade\" (alvo)\n",
        "def definir_prioridade(row):\n",
        "    score = 0\n",
        "    if row[\"tipo\"] in [\"Homicídio (tentativa)\", \"Tráfico\", \"Violência Doméstica\", \"Roubo\"]:\n",
        "        score += 2\n",
        "    if row[\"tem_arma\"] == 1:\n",
        "        score += 2\n",
        "    if row[\"vitima_ferida\"] == 1:\n",
        "        score += 2\n",
        "    if row[\"historico_reincidencia\"] == 1:\n",
        "        score += 1\n",
        "\n",
        "    if score >= 5:\n",
        "        return \"Alta\"\n",
        "    elif score >= 3:\n",
        "        return \"Média\"\n",
        "    else:\n",
        "        return \"Baixa\"\n",
        "\n",
        "df[\"prioridade\"] = df.apply(definir_prioridade, axis=1)\n",
        "\n",
        "df.head(10)\n"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 363
        },
        "id": "NsAT7crtwOQ8",
        "outputId": "7ec97025-503b-4dc9-99cc-3195398b9211"
      },
      "execution_count": 95,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "                    tipo       local    periodo  tem_arma  vitima_ferida  \\\n",
              "0  Homicídio (tentativa)  Planaltina  Madrugada         0              1   \n",
              "1    Violência Doméstica   Asa Norte      Noite         0              1   \n",
              "2            Estelionato   Samambaia  Madrugada         1              1   \n",
              "3  Homicídio (tentativa)  Planaltina      Noite         0              0   \n",
              "4                 Ameaça  Taguatinga      Tarde         0              0   \n",
              "5            Estelionato  Taguatinga      Noite         0              0   \n",
              "6            Estelionato  Taguatinga      Tarde         0              1   \n",
              "7  Homicídio (tentativa)  Taguatinga      Noite         0              0   \n",
              "8                  Roubo  Taguatinga  Madrugada         0              0   \n",
              "9                 Ameaça  Planaltina      Noite         0              0   \n",
              "\n",
              "   historico_reincidencia                                         descricao  \\\n",
              "0                       0        Relato de agressão grave com risco à vida.   \n",
              "1                       1  Relato de agressão/ameaça em contexto doméstico.   \n",
              "2                       0    Relato de golpe por aplicativo, pix ou cartão.   \n",
              "3                       1        Relato de agressão grave com risco à vida.   \n",
              "4                       0         Relato de ameaça verbal ou por mensagens.   \n",
              "5                       0    Relato de golpe por aplicativo, pix ou cartão.   \n",
              "6                       0    Relato de golpe por aplicativo, pix ou cartão.   \n",
              "7                       0        Relato de agressão grave com risco à vida.   \n",
              "8                       1         Relato de subtração com ameaça/violência.   \n",
              "9                       0         Relato de ameaça verbal ou por mensagens.   \n",
              "\n",
              "  prioridade  \n",
              "0      Média  \n",
              "1       Alta  \n",
              "2      Média  \n",
              "3      Média  \n",
              "4      Baixa  \n",
              "5      Baixa  \n",
              "6      Baixa  \n",
              "7      Baixa  \n",
              "8      Média  \n",
              "9      Baixa  "
            ],
            "text/html": [
              "\n",
              "  <div id=\"df-71af5b4d-65d8-4c94-840f-075589da889c\" class=\"colab-df-container\">\n",
              "    <div>\n",
              "<style scoped>\n",
              "    .dataframe tbody tr th:only-of-type {\n",
              "        vertical-align: middle;\n",
              "    }\n",
              "\n",
              "    .dataframe tbody tr th {\n",
              "        vertical-align: top;\n",
              "    }\n",
              "\n",
              "    .dataframe thead th {\n",
              "        text-align: right;\n",
              "    }\n",
              "</style>\n",
              "<table border=\"1\" class=\"dataframe\">\n",
              "  <thead>\n",
              "    <tr style=\"text-align: right;\">\n",
              "      <th></th>\n",
              "      <th>tipo</th>\n",
              "      <th>local</th>\n",
              "      <th>periodo</th>\n",
              "      <th>tem_arma</th>\n",
              "      <th>vitima_ferida</th>\n",
              "      <th>historico_reincidencia</th>\n",
              "      <th>descricao</th>\n",
              "      <th>prioridade</th>\n",
              "    </tr>\n",
              "  </thead>\n",
              "  <tbody>\n",
              "    <tr>\n",
              "      <th>0</th>\n",
              "      <td>Homicídio (tentativa)</td>\n",
              "      <td>Planaltina</td>\n",
              "      <td>Madrugada</td>\n",
              "      <td>0</td>\n",
              "      <td>1</td>\n",
              "      <td>0</td>\n",
              "      <td>Relato de agressão grave com risco à vida.</td>\n",
              "      <td>Média</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>1</th>\n",
              "      <td>Violência Doméstica</td>\n",
              "      <td>Asa Norte</td>\n",
              "      <td>Noite</td>\n",
              "      <td>0</td>\n",
              "      <td>1</td>\n",
              "      <td>1</td>\n",
              "      <td>Relato de agressão/ameaça em contexto doméstico.</td>\n",
              "      <td>Alta</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>2</th>\n",
              "      <td>Estelionato</td>\n",
              "      <td>Samambaia</td>\n",
              "      <td>Madrugada</td>\n",
              "      <td>1</td>\n",
              "      <td>1</td>\n",
              "      <td>0</td>\n",
              "      <td>Relato de golpe por aplicativo, pix ou cartão.</td>\n",
              "      <td>Média</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>3</th>\n",
              "      <td>Homicídio (tentativa)</td>\n",
              "      <td>Planaltina</td>\n",
              "      <td>Noite</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>1</td>\n",
              "      <td>Relato de agressão grave com risco à vida.</td>\n",
              "      <td>Média</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>4</th>\n",
              "      <td>Ameaça</td>\n",
              "      <td>Taguatinga</td>\n",
              "      <td>Tarde</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>Relato de ameaça verbal ou por mensagens.</td>\n",
              "      <td>Baixa</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>5</th>\n",
              "      <td>Estelionato</td>\n",
              "      <td>Taguatinga</td>\n",
              "      <td>Noite</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>Relato de golpe por aplicativo, pix ou cartão.</td>\n",
              "      <td>Baixa</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>6</th>\n",
              "      <td>Estelionato</td>\n",
              "      <td>Taguatinga</td>\n",
              "      <td>Tarde</td>\n",
              "      <td>0</td>\n",
              "      <td>1</td>\n",
              "      <td>0</td>\n",
              "      <td>Relato de golpe por aplicativo, pix ou cartão.</td>\n",
              "      <td>Baixa</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>7</th>\n",
              "      <td>Homicídio (tentativa)</td>\n",
              "      <td>Taguatinga</td>\n",
              "      <td>Noite</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>Relato de agressão grave com risco à vida.</td>\n",
              "      <td>Baixa</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>8</th>\n",
              "      <td>Roubo</td>\n",
              "      <td>Taguatinga</td>\n",
              "      <td>Madrugada</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>1</td>\n",
              "      <td>Relato de subtração com ameaça/violência.</td>\n",
              "      <td>Média</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>9</th>\n",
              "      <td>Ameaça</td>\n",
              "      <td>Planaltina</td>\n",
              "      <td>Noite</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>Relato de ameaça verbal ou por mensagens.</td>\n",
              "      <td>Baixa</td>\n",
              "    </tr>\n",
              "  </tbody>\n",
              "</table>\n",
              "</div>\n",
              "    <div class=\"colab-df-buttons\">\n",
              "\n",
              "  <div class=\"colab-df-container\">\n",
              "    <button class=\"colab-df-convert\" onclick=\"convertToInteractive('df-71af5b4d-65d8-4c94-840f-075589da889c')\"\n",
              "            title=\"Convert this dataframe to an interactive table.\"\n",
              "            style=\"display:none;\">\n",
              "\n",
              "  <svg xmlns=\"http://www.w3.org/2000/svg\" height=\"24px\" viewBox=\"0 -960 960 960\">\n",
              "    <path d=\"M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z\"/>\n",
              "  </svg>\n",
              "    </button>\n",
              "\n",
              "  <style>\n",
              "    .colab-df-container {\n",
              "      display:flex;\n",
              "      gap: 12px;\n",
              "    }\n",
              "\n",
              "    .colab-df-convert {\n",
              "      background-color: #E8F0FE;\n",
              "      border: none;\n",
              "      border-radius: 50%;\n",
              "      cursor: pointer;\n",
              "      display: none;\n",
              "      fill: #1967D2;\n",
              "      height: 32px;\n",
              "      padding: 0 0 0 0;\n",
              "      width: 32px;\n",
              "    }\n",
              "\n",
              "    .colab-df-convert:hover {\n",
              "      background-color: #E2EBFA;\n",
              "      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);\n",
              "      fill: #174EA6;\n",
              "    }\n",
              "\n",
              "    .colab-df-buttons div {\n",
              "      margin-bottom: 4px;\n",
              "    }\n",
              "\n",
              "    [theme=dark] .colab-df-convert {\n",
              "      background-color: #3B4455;\n",
              "      fill: #D2E3FC;\n",
              "    }\n",
              "\n",
              "    [theme=dark] .colab-df-convert:hover {\n",
              "      background-color: #434B5C;\n",
              "      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);\n",
              "      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));\n",
              "      fill: #FFFFFF;\n",
              "    }\n",
              "  </style>\n",
              "\n",
              "    <script>\n",
              "      const buttonEl =\n",
              "        document.querySelector('#df-71af5b4d-65d8-4c94-840f-075589da889c button.colab-df-convert');\n",
              "      buttonEl.style.display =\n",
              "        google.colab.kernel.accessAllowed ? 'block' : 'none';\n",
              "\n",
              "      async function convertToInteractive(key) {\n",
              "        const element = document.querySelector('#df-71af5b4d-65d8-4c94-840f-075589da889c');\n",
              "        const dataTable =\n",
              "          await google.colab.kernel.invokeFunction('convertToInteractive',\n",
              "                                                    [key], {});\n",
              "        if (!dataTable) return;\n",
              "\n",
              "        const docLinkHtml = 'Like what you see? Visit the ' +\n",
              "          '<a target=\"_blank\" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'\n",
              "          + ' to learn more about interactive tables.';\n",
              "        element.innerHTML = '';\n",
              "        dataTable['output_type'] = 'display_data';\n",
              "        await google.colab.output.renderOutput(dataTable, element);\n",
              "        const docLink = document.createElement('div');\n",
              "        docLink.innerHTML = docLinkHtml;\n",
              "        element.appendChild(docLink);\n",
              "      }\n",
              "    </script>\n",
              "  </div>\n",
              "\n",
              "\n",
              "    </div>\n",
              "  </div>\n"
            ],
            "application/vnd.google.colaboratory.intrinsic+json": {
              "type": "dataframe",
              "variable_name": "df",
              "summary": "{\n  \"name\": \"df\",\n  \"rows\": 200,\n  \"fields\": [\n    {\n      \"column\": \"tipo\",\n      \"properties\": {\n        \"dtype\": \"category\",\n        \"num_unique_values\": 7,\n        \"samples\": [\n          \"Homic\\u00eddio (tentativa)\",\n          \"Viol\\u00eancia Dom\\u00e9stica\",\n          \"Tr\\u00e1fico\"\n        ],\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    },\n    {\n      \"column\": \"local\",\n      \"properties\": {\n        \"dtype\": \"category\",\n        \"num_unique_values\": 7,\n        \"samples\": [\n          \"Planaltina\",\n          \"Asa Norte\",\n          \"Asa Sul\"\n        ],\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    },\n    {\n      \"column\": \"periodo\",\n      \"properties\": {\n        \"dtype\": \"category\",\n        \"num_unique_values\": 4,\n        \"samples\": [\n          \"Noite\",\n          \"Manh\\u00e3\",\n          \"Madrugada\"\n        ],\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    },\n    {\n      \"column\": \"tem_arma\",\n      \"properties\": {\n        \"dtype\": \"number\",\n        \"std\": 0,\n        \"min\": 0,\n        \"max\": 1,\n        \"num_unique_values\": 2,\n        \"samples\": [\n          1,\n          0\n        ],\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    },\n    {\n      \"column\": \"vitima_ferida\",\n      \"properties\": {\n        \"dtype\": \"number\",\n        \"std\": 0,\n        \"min\": 0,\n        \"max\": 1,\n        \"num_unique_values\": 2,\n        \"samples\": [\n          0,\n          1\n        ],\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    },\n    {\n      \"column\": \"historico_reincidencia\",\n      \"properties\": {\n        \"dtype\": \"number\",\n        \"std\": 0,\n        \"min\": 0,\n        \"max\": 1,\n        \"num_unique_values\": 2,\n        \"samples\": [\n          1,\n          0\n        ],\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    },\n    {\n      \"column\": \"descricao\",\n      \"properties\": {\n        \"dtype\": \"category\",\n        \"num_unique_values\": 7,\n        \"samples\": [\n          \"Relato de agress\\u00e3o grave com risco \\u00e0 vida.\",\n          \"Relato de agress\\u00e3o/amea\\u00e7a em contexto dom\\u00e9stico.\"\n        ],\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    },\n    {\n      \"column\": \"prioridade\",\n      \"properties\": {\n        \"dtype\": \"category\",\n        \"num_unique_values\": 3,\n        \"samples\": [\n          \"M\\u00e9dia\",\n          \"Alta\"\n        ],\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    }\n  ]\n}"
            }
          },
          "metadata": {},
          "execution_count": 95
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "Treinar um modelo de ML para prever prioridade"
      ],
      "metadata": {
        "id": "_TUApGn8xxem"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "from sklearn.model_selection import train_test_split\n",
        "from sklearn.preprocessing import OneHotEncoder\n",
        "from sklearn.compose import ColumnTransformer\n",
        "from sklearn.pipeline import Pipeline\n",
        "from sklearn.metrics import classification_report, confusion_matrix\n",
        "from sklearn.ensemble import RandomForestClassifier\n",
        "\n",
        "X = df[[\"tipo\",\"local\",\"periodo\",\"tem_arma\",\"vitima_ferida\",\"historico_reincidencia\"]]\n",
        "y = df[\"prioridade\"]\n",
        "\n",
        "# colunas categóricas e numéricas\n",
        "cat_cols = [\"tipo\",\"local\",\"periodo\"]\n",
        "num_cols = [\"tem_arma\",\"vitima_ferida\",\"historico_reincidencia\"]\n",
        "\n",
        "preprocess = ColumnTransformer(\n",
        "    transformers=[\n",
        "        (\"cat\", OneHotEncoder(handle_unknown=\"ignore\"), cat_cols),\n",
        "        (\"num\", \"passthrough\", num_cols)\n",
        "    ]\n",
        ")\n",
        "\n",
        "model = RandomForestClassifier(n_estimators=200, random_state=42)\n",
        "\n",
        "pipe = Pipeline(steps=[\n",
        "    (\"prep\", preprocess),\n",
        "    (\"model\", model)\n",
        "])\n",
        "\n",
        "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)\n",
        "\n",
        "pipe.fit(X_train, y_train)\n",
        "pred = pipe.predict(X_test)\n",
        "\n",
        "print(classification_report(y_test, pred))\n",
        "print(\"Matriz de confusão:\\n\", confusion_matrix(y_test, pred))\n"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "mIA-UsXVx13U",
        "outputId": "e60b03e1-0a77-4b7b-bd77-c1ec8155add8"
      },
      "execution_count": 96,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "              precision    recall  f1-score   support\n",
            "\n",
            "        Alta       0.60      0.75      0.67         4\n",
            "       Baixa       0.89      1.00      0.94        24\n",
            "       Média       0.88      0.58      0.70        12\n",
            "\n",
            "    accuracy                           0.85        40\n",
            "   macro avg       0.79      0.78      0.77        40\n",
            "weighted avg       0.86      0.85      0.84        40\n",
            "\n",
            "Matriz de confusão:\n",
            " [[ 3  0  1]\n",
            " [ 0 24  0]\n",
            " [ 2  3  7]]\n"
          ]
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "Testar com uma ocorrência “nova”"
      ],
      "metadata": {
        "id": "Lwy2be2gx-mt"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "nova = pd.DataFrame([{\n",
        "    \"tipo\": \"Violência Doméstica\",\n",
        "    \"local\": \"Ceilândia\",\n",
        "    \"periodo\": \"Noite\",\n",
        "    \"tem_arma\": 1,\n",
        "    \"vitima_ferida\": 0,\n",
        "    \"historico_reincidencia\": 1\n",
        "}])\n",
        "\n",
        "pipe.predict(nova)[0]\n"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 35
        },
        "id": "zc7nlW12yB6B",
        "outputId": "70b922e4-7335-48fa-9108-05c48b9276ec"
      },
      "execution_count": 97,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "'Média'"
            ],
            "application/vnd.google.colaboratory.intrinsic+json": {
              "type": "string"
            }
          },
          "metadata": {},
          "execution_count": 97
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "Pergunta ??\n",
        "Por que isso tende a dar Alta?\n",
        "O que mudaria se “tem_arma=0” e “vitima_ferida=0”?\n"
      ],
      "metadata": {
        "id": "d_vKKN0y6qMw"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "nova = pd.DataFrame([{\n",
        "    \"tipo\": \"Violência Doméstica\",\n",
        "    \"local\": \"Ceilândia\",\n",
        "    \"periodo\": \"Noite\",\n",
        "    \"tem_arma\": 0,\n",
        "    \"vitima_ferida\": 1,\n",
        "    \"historico_reincidencia\": 1\n",
        "}])\n",
        "\n",
        "pipe.predict(nova)[0]"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 35
        },
        "id": "fo1Rx2P46zOp",
        "outputId": "7139234f-bf63-478b-ef95-c926c59dd7bd"
      },
      "execution_count": 98,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "'Alta'"
            ],
            "application/vnd.google.colaboratory.intrinsic+json": {
              "type": "string"
            }
          },
          "metadata": {},
          "execution_count": 98
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "Mini-RAG de procedimentos"
      ],
      "metadata": {
        "id": "XeNlHlBHyTkH"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "kb = [\n",
        "    {\n",
        "        \"tema\": \"Preservação de Local\",\n",
        "        \"conteudo\": \"Em ocorrências com risco à vida ou crime grave, orientar a preservação do local e acionar equipe competente. Evitar contaminação de vestígios.\"\n",
        "    },\n",
        "    {\n",
        "        \"tema\": \"Violência Doméstica\",\n",
        "        \"conteudo\": \"Priorizar a segurança da vítima, avaliar risco imediato, orientar registro e medidas protetivas conforme protocolos vigentes.\"\n",
        "    },\n",
        "    {\n",
        "        \"tema\": \"Estelionato\",\n",
        "        \"conteudo\": \"Coletar evidências digitais (comprovantes, prints, contas), orientar preservação de registros e canais formais para bloqueio/contestação quando aplicável.\"\n",
        "    },\n",
        "    {\n",
        "        \"tema\": \"Ameaça\",\n",
        "        \"conteudo\": \"Registrar circunstâncias, identificar meio (presencial/mensagem), avaliar risco e orientar preservação de evidências (mensagens, áudios).\"\n",
        "    }\n",
        "]\n"
      ],
      "metadata": {
        "id": "BWB9vDG7yXEZ"
      },
      "execution_count": 99,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "source": [
        "Recuperação (buscar o trecho relevante)"
      ],
      "metadata": {
        "id": "o_ZalB0hyckH"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "def recuperar_contexto(pergunta, kb):\n",
        "    p = pergunta.lower()\n",
        "    hits = []\n",
        "    for item in kb:\n",
        "        if item[\"tema\"].lower() in p:\n",
        "            hits.append(item[\"conteudo\"])\n",
        "    if not hits:\n",
        "        # fallback simples: pega 1 conteúdo “mais geral”\n",
        "        hits = [kb[0][\"conteudo\"]]\n",
        "    return \"\\n\".join(hits)\n",
        "\n",
        "pergunta = \"Como agir em um caso de Violência Doméstica?\"\n",
        "contexto = recuperar_contexto(pergunta, kb)\n",
        "contexto\n"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 35
        },
        "id": "xRp3pDGEye31",
        "outputId": "937eaeb4-8847-4a96-e3f7-4a2d88022032"
      },
      "execution_count": 100,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "'Priorizar a segurança da vítima, avaliar risco imediato, orientar registro e medidas protetivas conforme protocolos vigentes.'"
            ],
            "application/vnd.google.colaboratory.intrinsic+json": {
              "type": "string"
            }
          },
          "metadata": {},
          "execution_count": 100
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "Geração (simulada) prompt engineering"
      ],
      "metadata": {
        "id": "Vj-GdaC3yiyF"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "prompt = f\"\"\"\n",
        "Você é um assistente de triagem e orientação.\n",
        "Use APENAS o contexto abaixo para responder, sem inventar.\n",
        "Se não houver informação, diga: \"Não consta na base\".\n",
        "CONTEXTO:\n",
        "{contexto}\n",
        "\n",
        "PERGUNTA:\n",
        "{pergunta}\n",
        "\n",
        "RESPOSTA:\n",
        "\"\"\"\n",
        "print(prompt)\n"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "WXrFjpBDylrT",
        "outputId": "70d01cb1-3002-4207-8ef1-2cae9e61ca13"
      },
      "execution_count": 101,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "\n",
            "Você é um assistente de triagem e orientação.\n",
            "Use APENAS o contexto abaixo para responder, sem inventar.\n",
            "Se não houver informação, diga: \"Não consta na base\".\n",
            "CONTEXTO:\n",
            "Priorizar a segurança da vítima, avaliar risco imediato, orientar registro e medidas protetivas conforme protocolos vigentes.\n",
            "\n",
            "PERGUNTA:\n",
            "Como agir em um caso de Violência Doméstica?\n",
            "\n",
            "RESPOSTA:\n",
            "\n"
          ]
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "Conectar ML + RAG em uma “triagem completa”"
      ],
      "metadata": {
        "id": "rn0Hf0CcyoGt"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "def triagem_completa(ocorrencia_dict, pergunta_orientacao):\n",
        "    # 1) prioridade via ML\n",
        "    o = pd.DataFrame([ocorrencia_dict])\n",
        "    prioridade = pipe.predict(o)[0]\n",
        "\n",
        "    # 2) contexto via RAG simples\n",
        "    contexto = recuperar_contexto(pergunta_orientacao, kb)\n",
        "\n",
        "    return {\n",
        "        \"prioridade_prevista\": prioridade,\n",
        "        \"contexto_recuperado\": contexto\n",
        "    }\n",
        "\n",
        "resultado = triagem_completa(\n",
        "    {\n",
        "        \"tipo\": \"Estelionato\",\n",
        "        \"local\": \"Asa Sul\",\n",
        "        \"periodo\": \"Manhã\",\n",
        "        \"tem_arma\": 0,\n",
        "        \"vitima_ferida\": 0,\n",
        "        \"historico_reincidencia\": 0\n",
        "    },\n",
        "    \"Quais evidências coletar em Estelionato?\"\n",
        ")\n",
        "\n",
        "resultado\n"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "Q6u-cElvyo3F",
        "outputId": "a98d64bd-8bfc-4236-bc7d-7737c82e647d"
      },
      "execution_count": 102,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "{'prioridade_prevista': 'Baixa',\n",
              " 'contexto_recuperado': 'Coletar evidências digitais (comprovantes, prints, contas), orientar preservação de registros e canais formais para bloqueio/contestação quando aplicável.'}"
            ]
          },
          "metadata": {},
          "execution_count": 102
        }
      ]
    }
  ]
}
