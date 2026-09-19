"""Gini, nullmodell-sammenligning og parvis bootstrap av modellforskjeller.

Alle modeller skåres på de samme radene. Bootstrapen trekker forsikringstakere
(``insured_id``) med tilbakelegging, slik at flere poliseår for samme kunde følger
hverandre, og beregner alle mål på samme utvalg. Forskjeller mellom modeller er
dermed parvise og tar hensyn til at prediksjonene henger sammen.

Gini er basert på ordnet Lorenz-kurve (Frees, Meyers og Cummings, 2011): rader
sorteres etter predikert risiko (lavest først), og kurven viser kumulativ
andel av skadekostnad mot kumulativ andel av eksponering.
"""

import numpy as np
import pandas as pd

NULL_MODEL = "Nullmodell"


def tweedie_unit_deviance(observed, prediction, power):
    """Tweedie-deviance per rad (uten vekt), for $1<p<2$."""
    observed = np.asarray(observed, dtype=float)
    prediction = np.asarray(prediction, dtype=float)
    return 2 * (
        observed ** (2 - power) / ((1 - power) * (2 - power))
        - observed * prediction ** (1 - power) / (1 - power)
        + prediction ** (2 - power) / (2 - power)
    )


def _tie_group_ends(sorted_score):
    """Indeks til siste rad i hver gruppe med lik score (likhet gir én kurvebit)."""
    return np.r_[np.flatnonzero(np.diff(sorted_score) != 0), len(sorted_score) - 1]


def lorenz_curve(cost, exposure, score):
    """Ordnet Lorenz-kurve: x = andel eksponering, y = andel skadekostnad."""
    order = np.argsort(score, kind="stable")
    ends = _tie_group_ends(score[order])
    x = np.r_[0, np.cumsum(exposure[order])[ends] / exposure.sum()]
    y = np.r_[0, np.cumsum(cost[order])[ends] / cost.sum()]
    return x, y


def gini_index(x, y):
    """Gini = 1 − 2·(areal under Lorenz-kurven); 0 for konstant score."""
    return 1 - np.sum(np.diff(x) * (y[1:] + y[:-1]))


def _prepare_models(observed, predictions, power):
    """Forhåndsberegn radvis deviance og rangering, som er felles for alle utvalg."""
    prepared = {}
    for name, prediction in predictions.items():
        values = prediction.to_numpy(dtype=float)
        order = np.argsort(values, kind="stable")
        prepared[name] = {
            "prediction": values,
            "unit_deviance": tweedie_unit_deviance(observed, values, power),
            "order": order,
            "ends": _tie_group_ends(values[order]),
        }
    return prepared


def _metrics(prepared, multiplicity, exposure, cost):
    """Deviance, Gini og balanse for hvert utvalg (rad i ``multiplicity``).

    ``multiplicity[b, i]`` er hvor mange ganger rad ``i`` er trukket i utvalg ``b``.
    Rangeringen er fast, så Gini krever ingen ny sortering per utvalg.
    """
    weight = multiplicity * exposure
    loss = multiplicity * cost
    total_weight, total_loss = weight.sum(axis=1), loss.sum(axis=1)
    zeros = np.zeros((len(weight), 1))
    result = {"deviance": {}, "gini": {}, "balance": {}}
    for name, model in prepared.items():
        result["deviance"][name] = (weight * model["unit_deviance"]).sum(
            axis=1
        ) / total_weight
        result["balance"][name] = (weight * model["prediction"]).sum(
            axis=1
        ) / total_loss - 1
        order, ends = model["order"], model["ends"]
        x = np.hstack([zeros, np.cumsum(weight[:, order], axis=1)[:, ends]])
        y = np.hstack([zeros, np.cumsum(loss[:, order], axis=1)[:, ends]])
        x, y = x / total_weight[:, None], y / total_loss[:, None]
        gini = 1 - (np.diff(x, axis=1) * (y[:, 1:] + y[:, :-1])).sum(axis=1)
        result["gini"][name] = (
            np.round(gini, 12) + 0.0
        )  # fjerner -0.0 og avrundingsstøy
    return result


def paired_bootstrap(
    test_frame, predictions, power, *, n_boot=5000, seed=100, batch_size=200
):
    """Parvis bootstrap over forsikringstakere.

    Returnerer ``{"point": ..., "replicates": ...}`` med hvert av målene
    ``deviance``, ``gini`` og ``balance`` (balanse som andel, ikke prosent).
    ``point`` er verdiene på det opprinnelige utvalget, ``replicates`` én rad per
    bootstrap-utvalg.
    """
    cost = test_frame["property_incurred"].to_numpy(dtype=float)
    exposure = test_frame["total_exposure"].to_numpy(dtype=float)
    prepared = _prepare_models(cost / exposure, predictions, power)

    once = _metrics(prepared, np.ones((1, len(cost))), exposure, cost)
    point = {
        metric: pd.Series({name: values[0] for name, values in per_model.items()})
        for metric, per_model in once.items()
    }

    codes, customers = pd.factorize(test_frame["insured_id"])
    n_customers = len(customers)
    rng = np.random.default_rng(seed)
    parts = {metric: {name: [] for name in prepared} for metric in point}
    for start in range(0, n_boot, batch_size):
        size = min(batch_size, n_boot - start)
        counts = rng.multinomial(
            n_customers, np.full(n_customers, 1 / n_customers), size
        )
        batch = _metrics(prepared, counts[:, codes], exposure, cost)
        for metric, per_model in batch.items():
            for name, values in per_model.items():
                parts[metric][name].append(values)
    replicates = {
        metric: pd.DataFrame({name: np.concatenate(v) for name, v in per_model.items()})
        for metric, per_model in parts.items()
    }
    return {"point": point, "replicates": replicates}


def _interval(samples, level=0.95):
    """Percentilintervall for hver kolonne."""
    tail = (1 - level) / 2
    return samples.quantile(tail), samples.quantile(1 - tail)


def build_model_summary(results, test_frame, boot, null_name=NULL_MODEL):
    """Én rad per modell: deviance, $D^2$ mot nullmodellen, Gini, MAE og balanse.

    Punktestimatene kommer fra hele testutvalget; intervallene er 95 %
    percentilintervall fra bootstrapen. $D^2 = 1 - D_{\\text{modell}} / D_{\\text{null}}$.
    """
    point, replicates = boot["point"], boot["replicates"]
    d2_point = 1 - point["deviance"] / point["deviance"][null_name]
    d2_samples = 1 - replicates["deviance"].div(
        replicates["deviance"][null_name], axis=0
    )
    gini_lo, gini_hi = _interval(replicates["gini"])
    d2_lo, d2_hi = _interval(d2_samples)
    balance_lo, balance_hi = _interval(replicates["balance"] * 100)

    cost = test_frame["property_incurred"].to_numpy(dtype=float)
    exposure = test_frame["total_exposure"].to_numpy(dtype=float)
    perfect = gini_index(*lorenz_curve(cost, exposure, cost / exposure))

    summary = results.set_index("Modell")
    table = pd.DataFrame(
        {
            "deviance": summary["Tweedie-deviance"],
            "d2": 100 * d2_point,
            "d2_lo": 100 * d2_lo,
            "d2_hi": 100 * d2_hi,
            "gini": point["gini"],
            "gini_lo": gini_lo,
            "gini_hi": gini_hi,
            "gini_normalized": point["gini"] / perfect,
            "mae": summary["Vektet MAE"],
            "balance": 100 * point["balance"],
            "balance_lo": balance_lo,
            "balance_hi": balance_hi,
        }
    )
    order = [null_name] + [n for n in results["Modell"] if n != null_name]
    return table.loc[order]


def format_model_summary(table):
    """Norsk visningstabell med intervaller i hakeparentes."""

    def with_interval(value, low, high, digits):
        if low == high:  # nullmodellen: referansen, ingen usikkerhet
            return f"{value:.{digits}f} (referanse)"
        return f"{value:.{digits}f} [{low:.{digits}f}, {high:.{digits}f}]"

    return pd.DataFrame(
        {
            "Devians": table["deviance"].map("{:.3f}".format),
            "D² i % [95 % KI]": [
                with_interval(r.d2, r.d2_lo, r.d2_hi, 2) for r in table.itertuples()
            ],
            "Gini [95 % KI]": [
                with_interval(r.gini, r.gini_lo, r.gini_hi, 3)
                for r in table.itertuples()
            ],
            "Normalisert Gini": table["gini_normalized"].map("{:.3f}".format),
            "Vektet MAE": table["mae"].map("{:.1f}".format),
            "Balanse i % [95 % KI]": [
                with_interval(r.balance, r.balance_lo, r.balance_hi, 1)
                for r in table.itertuples()
            ],
        }
    )


def build_pairwise_table(boot, pairs):
    """Parvise forskjeller ``(A, B)`` der positivt tall betyr at A er bedre.

    Devians-gevinst er $D_B - D_A$ (lavere devians er bedre), Gini-gevinst er
    $G_A - G_B$. ``p_better`` er andelen bootstrap-utvalg der A er bedre.
    """
    point, replicates = boot["point"], boot["replicates"]
    rows = []
    for a, b in pairs:
        row = {"pair": f"{a} mot {b}"}
        gains = {
            "deviance": (
                point["deviance"][b] - point["deviance"][a],
                replicates["deviance"][b] - replicates["deviance"][a],
            ),
            "gini": (
                point["gini"][a] - point["gini"][b],
                replicates["gini"][a] - replicates["gini"][b],
            ),
        }
        for metric, (estimate, samples) in gains.items():
            low, high = samples.quantile([0.025, 0.975])
            row.update(
                {
                    f"{metric}_gain": estimate,
                    f"{metric}_lo": low,
                    f"{metric}_hi": high,
                    f"{metric}_p_better": np.nan
                    if samples.abs().max() == 0
                    else (samples > 0).mean(),
                }
            )
        rows.append(row)
    return pd.DataFrame(rows).set_index("pair")


def format_pairwise_table(pairwise):
    """Norsk visningstabell for parvise forskjeller."""

    def cell(row, metric, digits):
        gain, low, high = (row[f"{metric}_{k}"] for k in ("gain", "lo", "hi"))
        return f"{gain:+.{digits}f} [{low:+.{digits}f}, {high:+.{digits}f}]"

    return pd.DataFrame(
        {
            "Devians-gevinst [95 % KI]": [
                cell(r, "deviance", 3) for _, r in pairwise.iterrows()
            ],
            "P(A bedre), devians": pairwise["deviance_p_better"].map(
                lambda p: "–" if np.isnan(p) else f"{p:.0%}"
            ),
            "Gini-gevinst [95 % KI]": [
                cell(r, "gini", 4) for _, r in pairwise.iterrows()
            ],
            "P(A bedre), Gini": pairwise["gini_p_better"].map(
                lambda p: "–" if np.isnan(p) else f"{p:.0%}"
            ),
        }
    )
