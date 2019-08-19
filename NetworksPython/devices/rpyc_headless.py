import rpyc.utils.classic
from rpyc.utils.server import OneShotServer
import CtxCtlTools
import CtxDynapse

# - Get list of devices
CtxCtlTools.device_controller.refresh_devices()
lDevices = CtxCtlTools.device_controller.get_unopened_devices()

# - Check that a device is available
assert lDevices is not None, "No devices found."

# - Open the first available device (or find the one we are interested in opening)
CtxCtlTools.device_controller.open_device(lDevices[0])

# - Wait until the device model has been created
while not hasattr(CtxDynapse, "model"):
    CtxCtlTools.process_events()

c = rpyc.utils.classic.SlaveService()
t = OneShotServer(c, port=1300)

print("RPyC: Ready to start.")
t.start()
