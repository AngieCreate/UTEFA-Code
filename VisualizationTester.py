import pandas as pd
import matplotlib.pyplot as plt

# Load Dataset and Set Variables
# file_path = "UTEFA_QuantiFi_Contestant_Dataset.csv"
# file_path = "scenario_crash_recovery.csv"
file_path = "scenario_flat.csv"
# file_path = "scenario_trending.csv"
# file_path = "scenario_volatile.csv"
df = pd.read_csv(file_path)

stocks = ["Stock_A", "Stock_B", "Stock_C", "Stock_D", "Stock_E"]
shortEMA = 20
largeEMA = 100

# Calculate EMAs
for stock in stocks:
    df[f"{stock}_EMA{shortEMA}"] = df[stock].ewm(span=shortEMA, adjust=False).mean()
    df[f"{stock}_EMA{largeEMA}"] = df[stock].ewm(span=largeEMA, adjust=False).mean()

# === Indicators for EACH STOCK ===
for stock in stocks:
    df[f"{stock}_Buy"] = (
        (df[f"{stock}_EMA{shortEMA}"].shift(1) < df[f"{stock}_EMA{largeEMA}"].shift(1)) &
        (df[f"{stock}_EMA{shortEMA}"] > df[f"{stock}_EMA{largeEMA}"])
    )

    df[f"{stock}_Sell"] = (
        (df[f"{stock}_EMA{shortEMA}"].shift(1) > df[f"{stock}_EMA{largeEMA}"].shift(1)) &
        (df[f"{stock}_EMA{shortEMA}"] < df[f"{stock}_EMA{largeEMA}"])
    )

    threshold = df[stock] * 0.001
    df[f"{stock}_Touch"] = abs(df[stock] - df[f"{stock}_EMA{largeEMA}"]) < threshold

    df[f"{stock}_Support_EMA{largeEMA}"] = (
        (df[stock].shift(1) > df[f"{stock}_EMA{largeEMA}"].shift(1)) &
        df[f"{stock}_Touch"] &
        (df[stock].shift(-1) > df[f"{stock}_EMA{largeEMA}"].shift(-1))
    )

    df[f"{stock}_Resistance_EMA{largeEMA}"] = (
        (df[stock].shift(1) < df[f"{stock}_EMA{largeEMA}"].shift(1)) &
        df[f"{stock}_Touch"] &
        (df[stock].shift(-1) < df[f"{stock}_EMA{largeEMA}"].shift(-1))
    )

# --- Create Subplots ---
fig, axes = plt.subplots(5, 1, figsize=(20, 20), sharex=True)

# ----- Create 1 legend ONLY -----
# Store only ONE example of each plotted style
legend_handles = []
legend_labels = []
legend_added = set()

for i, stock in enumerate(stocks):
    ax = axes[i]

    # Price
    h_price, = ax.plot(df["Day"], df[stock], label="Price", linewidth=1.5)
    if "Price" not in legend_added:
        legend_handles.append(h_price)
        legend_labels.append("Price")
        legend_added.add("Price")

    # EMA Short
    h_ema_s, = ax.plot(df["Day"], df[f"{stock}_EMA{shortEMA}"], linestyle="--", label=f"EMA{shortEMA}")
    if f"EMA{shortEMA}" not in legend_added:
        legend_handles.append(h_ema_s)
        legend_labels.append(f"EMA{shortEMA}")
        legend_added.add(f"EMA{shortEMA}")

    # EMA Long
    h_ema_l, = ax.plot(df["Day"], df[f"{stock}_EMA{largeEMA}"], linestyle=":", label=f"EMA{largeEMA}")
    if f"EMA{largeEMA}" not in legend_added:
        legend_handles.append(h_ema_l)
        legend_labels.append(f"EMA{largeEMA}")
        legend_added.add(f"EMA{largeEMA}")

    # BUY
    buy_idx = df[f"{stock}_Buy"]
    h_buy = ax.scatter(
        df.loc[buy_idx, "Day"],
        df.loc[buy_idx, stock],
        color="green", s=80
    )
    if "BUY" not in legend_added:
        legend_handles.append(h_buy)
        legend_labels.append("BUY")
        legend_added.add("BUY")

    # SELL
    sell_idx = df[f"{stock}_Sell"]
    h_sell = ax.scatter(
        df.loc[sell_idx, "Day"],
        df.loc[sell_idx, stock],
        color="red", s=80
    )
    if "SELL" not in legend_added:
        legend_handles.append(h_sell)
        legend_labels.append("SELL")
        legend_added.add("SELL")

    # SUPPORT
    support_idx = df[f"{stock}_Support_EMA{largeEMA}"]
    h_sup = ax.scatter(
        df.loc[support_idx, "Day"],
        df.loc[support_idx, stock],
        color="lightgreen", s=60
    )
    if "Support" not in legend_added:
        legend_handles.append(h_sup)
        legend_labels.append("Support")
        legend_added.add("Support")

    # RESISTANCE
    resist_idx = df[f"{stock}_Resistance_EMA{largeEMA}"]
    h_res = ax.scatter(
        df.loc[resist_idx, "Day"],
        df.loc[resist_idx, stock],
        color="pink", s=60
    )
    if "Resistance" not in legend_added:
        legend_handles.append(h_res)
        legend_labels.append("Resistance")
        legend_added.add("Resistance")

    ax.set_ylabel(stock)
    ax.grid(True, linestyle="--", alpha=0.6)

axes[-1].set_xlabel("Day")

fig.suptitle(f"EMA{shortEMA} & EMA{largeEMA} for 5 Stocks for dataset: {file_path}", fontsize=18)

# Tighten layout but keep space for legend
plt.subplots_adjust(right=0.78)

# Insert ONE combined legend
fig.legend(
    legend_handles,
    legend_labels,
    loc="center left",
    bbox_to_anchor=(0.82, 0.5),
    frameon=True
)

plt.tight_layout(rect=[0, 0, 0.80, 0.96])
plt.show()