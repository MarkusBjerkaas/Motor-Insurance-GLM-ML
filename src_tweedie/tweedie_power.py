"""Tweedie-power $p$ utledet fra severity-dispersjonen (T-05)."""

import statsmodels.api as sm

from src_core_glm.glm_core import fit_glm, glm_spec, prepare_design_frame


def severity_implied_power(severity_frame, terms):
    """Estimer $p$ fra en Gamma-GLM på snittskade, vektet med antall skader.

    En Tweedie med $1<p<2$ er en sammensatt Poisson–Gamma-variabel. Er
    skadestørrelsen Gamma med dispersjon $\\phi_s$ (= CV$^2$, shape
    $\\alpha = 1/\\phi_s$), er

    $$p = \\frac{\\alpha + 2}{\\alpha + 1} = \\frac{1 + 2\\phi_s}{1 + \\phi_s}$$

    $\\phi_s$ er Pearson-$\\chi^2$/df fra Gamma-modellen (``result.scale``).
    Gir alltid $1<p<2$ for $\\phi_s>0$. Forutsetter Gamma-fordelte skader med
    konstant CV, se limitation i notebooken.
    """
    design = prepare_design_frame(severity_frame, severity_frame, terms)
    spec = glm_spec(
        "severity_anker",
        terms,
        "average_severity",
        severity_frame,
        sm.families.Gamma(sm.families.links.Log()),
        "property_claims",
        2,  # Gamma = Tweedie med p = 2 (kun brukt til scoring, ikke her)
        required_columns=terms,
    )
    dispersion = fit_glm(spec, design).scale
    return {
        "dispersion": dispersion,
        "power": (1 + 2 * dispersion) / (1 + dispersion),
        "n_severity_rows": len(severity_frame),
    }
