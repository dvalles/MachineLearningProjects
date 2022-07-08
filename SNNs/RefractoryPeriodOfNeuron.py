"""
Neurons cannot continually fire, they have a refractory period of about 1ms between firings
This is caused by voltage gated ion channels that have trouble in that 1ms and therefore
The charge can't reset
"""

import time

#params
r_p = .001

#fire a neuron's action potential
def Fire(lastFire):
    since = time.time() - lastFire
    if since < r_p:
        time.sleep(r_p - since)

    print('Fired: ' + str(time.time()))
    return time.time()

lastFire = 0
lastFire = Fire(lastFire)
lastFire = Fire(lastFire)

