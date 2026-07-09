import numpy as np


def HermiteEnvelope(duration, amplitude, type):
    # Implements a Hermite pulse (no offset, with minimal length)
    if type not in ["pi", "halfpi"]:
        raise ValueError("Only types 'pi' or 'halfpi currently supported")
    
    eta = 0.956 if type == "pi" else 0.667

    fine_step = 203e-13 # FIXME: Instead of hardcoding, have this adapt to the actual timing

    t = np.arange(0, duration, fine_step)

    X = (t-duration/2) / (0.1667 * duration)

    envelope = amplitude * (1 - eta*X**2) * np.exp(-X**2)

    return envelope


# Alternative: 
# Just use coefficient modulation, issue, need a length  that is long enough for a single pi pulse.