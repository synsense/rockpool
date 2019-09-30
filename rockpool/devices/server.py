import rpyc.utils.classic
from rpyc.utils.server import OneShotServer

c = rpyc.utils.classic.SlaveService()
t = OneShotServer(c, port=1300)

print("RPyC: Ready to start.")
t.start()
