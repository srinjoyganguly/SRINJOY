from SimpleWebSocketServer import SimpleWebSocketServer, WebSocket
import json
import threading
import time as tm



# velocity variables to send to simulator
throttle = 0
steer = 0
brake = 0
state = 0
start_time = 0

class Bridge(object):
        def __init__(self):
                self.client = None
        def reg(self, client):
                self.client = client
                
bridge = Bridge()
                
class SimpleMove(WebSocket):
    
    def handleMessage(self):

        # modify velocity variables to send to simulator based on position values recieved by simulator
        global throttle, steer, brake, state, start_time

        s = str(self.data)

        my_bytes_value = self.data

        # Decode UTF-8 bytes to Unicode, and convert single quotes 
        # to double quotes to make it valid JSON
        my_json = my_bytes_value.decode('utf8').replace("'", '"')
        data = json.loads(my_json)
        
        # recieved position variables from simulator
        pos_x, pos_y, time, velocity = get_telemetry_data(data)

        # Execute student code and receive update simulated car parameters.
        controller = control(pos_x, pos_y, time, velocity)
        throttle, steer, brake  = (controller.get("throttle"), 
                                                               controller.get("steer"), 
                                                               controller.get("brake") 
                                                               )
        
        throttle = min(max(throttle,-1.0),1.0)
        throttle = 0.5*throttle
        
        # create json message and set velocity variable values
        msgJson = {"throttle": throttle, "steer":steer, "brake":brake}

        #send message
        self.sendMessage('{move:'+json.dumps(msgJson)+'}')
        
    def handleConnected(self):
        bridge.reg(self)
        print("CONNECTED")
        print(self.address, 'connected')
    def handleClose(self):
        print(self.address, 'closed')

def get_telemetry_data(data):
    """ Returns tuple of telemetry data"""
    pos_x = float(data["telemetry"]["pos_x"])
    pos_y = float(data["telemetry"]["pos_y"])
    time = float(data["telemetry"]["time"])
    velocity = float(data["telemetry"]["velocity"])
    return pos_x, pos_y, time, velocity
        
def run(controller):
    print("running",flush=True)
    global control
    control = controller
    overall_start_time = tm.time()
    server = SimpleWebSocketServer('0.0.0.0',3001, SimpleMove)
    t = threading.Thread(target=server.serveforever)
    t.daemon = True
    t.start()
    while 1: t.join(1)
        