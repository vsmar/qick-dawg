import os
os.environ['PYRO_SERIALIZER']              = 'pickle'
os.environ['PYRO_SERIALIZERS_ACCEPTED']    = 'pickle'
os.environ['PYRO_PICKLE_PROTOCOL_VERSION'] = '4'

import sys
import Pyro4
import qickdawg as qd

NS_HOST      = "192.168.3.1"
LASER_PMOD   = 0
LASER_ON_TUS = 2

Pyro4.config.SERIALIZER              = "pickle"
Pyro4.config.SERIALIZERS_ACCEPTED    = set(['pickle'])
Pyro4.config.PICKLE_PROTOCOL_VERSION = 4

def make_config():
    cfg = qd.NVConfiguration()
    cfg.laser_gate_pmod = LASER_PMOD
    cfg.laser_on_tus    = LASER_ON_TUS
    cfg.adc_channel = 0
    cfg.relax_delay_tus = 10
    return cfg

if __name__ == "__main__":
    cmd = sys.argv[1].lower() if len(sys.argv) > 1 else ""
    if cmd not in ("on", "off"):
        print("Usage: python laser_control.py [on|off]")
        sys.exit(1)

    try:
        qd.start_client(NS_HOST)
    except Exception as e:
        print(f"Connection failed: {e}")
        sys.exit(2)

    if cmd == "on":
        qd.laser_on(make_config())
    else:
        qd.laser_off(make_config())

    sys.exit(0)
