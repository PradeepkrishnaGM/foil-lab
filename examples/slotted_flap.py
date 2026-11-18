"""Two-element section: NACA 2412 main element with a slotted flap (a 30 % chord NACA 2412 below and
behind the main trailing edge, deflected 20 deg), inviscid and viscous at Re 3e6.

Run from the repository root:  python examples/slotted_flap.py   (about two minutes on one core)

The result is printed per element and in total, with the limits of the method next to it.
"""

import warnings

import foil_lab as fl

LIMITS = """Limits of this result (SPEC.md 4.6):
  * no merging of the main element's wake with the flap's boundary layer (the wake passes the
    flap as a separate layer, its displacement acting on the flap only through the potential flow);
  * no model of separation in the slot or of slot-flow effects beyond the potential flow;
  * no maximum-lift prediction for high-lift configurations;
  * not validated against XFOIL (single-element only) or experiment: consistency tests only
    (exact two-element potential flow, far-apart elements equal to single elements);
  * transition x is the global x coordinate (for the flap it is not x/c of the flap)."""


def configuration() -> fl.Configuration:
    main = fl.Airfoil.naca("2412")
    flap = fl.Airfoil.naca("2412")
    # scaled to 0.3 c about its leading edge, deflected 20 deg trailing edge down (nose-up positive
    # rotation of the element), leading edge at x = 0.97 c, 0.04 c below the chord line
    return fl.Configuration((fl.Element(main),
                             fl.Element(flap, scale=0.3, rotation_deg=20.0, pivot=(0.0, 0.0),
                                        offset=(0.97, -0.04))),
                            name="NACA 2412 with 30 % slotted flap at 20 deg")


def main() -> None:
    cfg = configuration()
    alphas = [0.0, 4.0, 8.0]
    print(f"{cfg.name}; reference chord {cfg.reference()[0]:.3f}, Cm about {cfg.reference()[1]}\n")
    print("Inviscid:")
    for r in fl.polar(cfg, alphas).results:
        parts = ", ".join(f"{e.name} {e.cl:+.3f}" for e in r.elements)
        print(f"  alpha {r.alpha:4.1f}  Cl {r.cl:+.3f}  ({parts})  Cm {r.cm:+.3f}")
    print("\nViscous, Re 3e6 (N_crit 9):")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pol = fl.polar(cfg, alphas, re=3e6)
    for r in pol.results:
        print(f"  alpha {r.alpha:4.1f}  converged {r.converged}  Cl {r.cl:+.3f}  Cd {r.cd:.4f}  "
              f"Cm {r.cm:+.3f}")
        for k, e in enumerate(r.elements):
            print(f"      element {k}: Cl {e.cl:+.3f}  Cd {e.cd:.4f} (friction {e.cd_friction:.4f})  "
                  f"x_tr upper {e.xtr_upper:.3f} lower {e.xtr_lower:.3f}")
    print()
    print(LIMITS)


if __name__ == "__main__":
    main()
