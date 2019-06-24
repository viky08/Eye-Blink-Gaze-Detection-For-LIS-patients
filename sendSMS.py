import time 
from time import sleep 
from sinchsms import SinchSMS 
  
# function for sending SMS 
class sendSMS():

    def __init__(self):

        self.number = '919250028074'
        self.app_key = 'e5c3efa0-3670-44bd-ab6b-1ab72db39a2e'
        self.app_secret = 'mJDlQUvxJEyDUoGMrLLxWg=='
        self.message = 'Need Help!!'
  
  
    
    def sendmessage(self,input):

        self.message = input
        client = SinchSMS(self.app_key, self.app_secret) 
        print("Sending '%s' to %s" % (self.message, self.number)) 
      
        response = client.send_message( self.number, self.message) 
        message_id = response['messageId'] 
        response = client.check_status(message_id) 
      
        # keep trying unless the status retured is Successful 
        while response['status'] != 'Successful': 
            print(response['status']) 
            # time.sleep(1) 
            response = client.check_status(message_id) 
      
        print(response['status']) 
  
