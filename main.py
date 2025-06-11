# --------------------------------------------
# Desafio – detecção de anomalias nos registros de consumo de materiais
# --------------------------------------------
import numpy as np
import pandas as pd
from scipy import stats
import plotly.express as px
import plotly.graph_objects as go

# ------------------------------------------------------------------
# 1. Simulação de registros manuais de entrada/saída de materiais
# ------------------------------------------------------------------
def simulate_data(
    n_units: int = 4,
    n_materials: int = 6,
    days: int = 120,
    seed: int = 42
) -> pd.DataFrame:
    """
    Cria um DataFrame de movimentações de materiais (entradas e saídas)
    para várias unidades operacionais ao longo de dias.
    """
    rng = np.random.default_rng(seed) # Remova a seed para aleatorizar os dados
    dates = pd.date_range(end=pd.Timestamp.today().normalize(), periods=days)

    units = [f"U{u+1:02d}" for u in range(n_units)]
    materials = [f"MAT-{m+1:03d}" for m in range(n_materials)]

    rows = []
    for date in dates:
        for unit in units:
            for mat in materials:
                # Entradas (positivas) e saídas (negativas)
                in_qty = rng.poisson(20)          # entrada média diária ~20
                out_qty = rng.poisson(18)         # saída média diária ~18
                rows.append(
                    [date, unit, mat, "entrada", in_qty]
                )
                rows.append(
                    [date, unit, mat, "saida", -out_qty]
                )

                # Injeta ruído anômalo em ~2% dos registros
                if rng.random() < 0.02:
                    anomaly = rng.integers(-150, 150)
                    rows.append([date, unit, mat, "saida", anomaly])

    df = pd.DataFrame(
        rows, columns=["data", "unidade", "material", "tipo", "qtd"]
    ).sort_values("data")

    # Adiciona custo unitário fixo por material
    cost_map = {mat: rng.uniform(5, 40) for mat in materials}
    df["custo_unit"] = df["material"].map(cost_map)
    df["valor_movto"] = df["qtd"] * df["custo_unit"]
    return df


# ------------------------------------------------------------------
# 2. Métricas de variação & detecção de outliers (Z-score e IQR)
# ------------------------------------------------------------------
def calc_zscore_flags(df: pd.DataFrame, z_thresh: float = 3.0) -> pd.Series:
    """
    Retorna um Series booleano indicando Z-scores fora do limite |z| > z_thresh,
    calculado por material + unidade.
    """
    zscores = (
        df.groupby(["unidade", "material"])["qtd"]
        .transform(lambda x: stats.zscore(x, nan_policy="omit"))
    )
    return (zscores.abs() > z_thresh).rename("flag_zscore")


def calc_iqr_flags(df: pd.DataFrame, k: float = 1.5) -> pd.Series:
    """
    IQR por material + unidade – flag se qtd < Q1 - k*IQR ou qtd > Q3 + k*IQR.
    """
    q1 = df.groupby(["unidade", "material"])["qtd"].transform(
        lambda x: x.quantile(0.25)
    )
    q3 = df.groupby(["unidade", "material"])["qtd"].transform(
        lambda x: x.quantile(0.75)
    )
    iqr = q3 - q1
    lower = q1 - k * iqr
    upper = q3 + k * iqr
    return ((df["qtd"] < lower) | (df["qtd"] > upper)).rename("flag_iqr")


# ------------------------------------------------------------------
# 3. Painel de alertas
# ------------------------------------------------------------------
def build_alert_panel(df: pd.DataFrame) -> pd.DataFrame:
    """
    Gera painel consolidado com registros fora do padrão, destacando
    desvios, sinais e impacto financeiro estimado.
    """
    df = df.copy()
    df["flag_zscore"] = calc_zscore_flags(df)
    df["flag_iqr"] = calc_iqr_flags(df)
    df["flag_any"] = df[["flag_zscore", "flag_iqr"]].any(axis=1)

    anomalies = df[df["flag_any"]].copy()

    # Calcula Z-score explícito para mostrar no painel
    anomalies["zscore"] = (
        df.groupby(["unidade", "material"])["qtd"]
        .transform(lambda x: stats.zscore(x, nan_policy="omit"))
    ).round(2)

    # Impacto monetário absoluto (quanto o movimento foge do zero)
    anomalies["impacto_R$"] = (anomalies["qtd"].abs() * anomalies["custo_unit"]).round(2)
    return anomalies[
        [
            "data",
            "unidade",
            "material",
            "tipo",
            "qtd",
            "zscore",
            "flag_zscore",
            "flag_iqr",
            "impacto_R$",
        ]
    ].sort_values("impacto_R$", ascending=False)


# ------------------------------------------------------------------
# 4. Simulação de impacto financeiro agregado
# ------------------------------------------------------------------
def estimate_financial_impact(anomalies: pd.DataFrame) -> float:
    """
    Retorna o somatório em reais do impacto estimado dos desvios.
    """
    return anomalies["impacto_R$"].sum().round(2)


# ------------------------------------------------------------------
# 5. Execução
# ------------------------------------------------------------------
if __name__ == "__main__":
    # 5.1 Gera dados
    df_registros = simulate_data()

    # 5.2 Painel de alertas
    painel = build_alert_panel(df_registros)
    print("\n=== PRIMEIROS ALERTAS DETECTADOS ===")
    print(painel.head(10).to_string(index=False))

    # 5.3 Visualização interativa simples
    fig = px.scatter(
        painel,
        x="data",
        y="qtd",
        color="material",
        symbol="unidade",
        hover_data=["zscore", "impacto_R$"],
        title="Dispersion plot – registros fora do padrão"
    )
    fig.update_traces(marker_size=9)
    fig.show()

    # 5.4 Resumo de impacto financeiro
    impacto_total = estimate_financial_impact(painel)
    print(f"\n>>> Impacto financeiro POTENCIAL dos desvios: R$ {impacto_total:,.2f}")

    # 5.5 Tabela interativa de alertas
    alert_table = go.Figure(
        data=[go.Table(
            header=dict(values=list(painel.columns),
                        fill_color="lightgrey",
                        align="left"),
            cells=dict(values=[painel[col] for col in painel.columns],
                       align="left"))
        ]
    )
    alert_table.update_layout(title_text="Painel de Alertas – Detalhado")
    # alert_table.show()
