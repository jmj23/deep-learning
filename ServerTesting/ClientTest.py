#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  9 09:45:48 2018

@author: jmj136
"""

import socket
import sys

# Create a TCP/IP socket
sock = socket.socket(socket.AF_INET,socket.SOCK_STREAM)

# Connect the socket to the port where the server is listening
server_address = ('localhost',10000)
print('Connection to %s port %s' % server_address,file=sys.stderr)
sock.connection(server_address)

try:
    # Send data
    message = 'This is the message. It will be repeated.'
    print('Sending "%s"' % message, file=sys.stderr)
    sock.sendall(message)
    
    # Look for the response
    amount_received = 0
    amount_expected = len(message)
    
    while amount_received < amount_expected:
        data = sock.recv(16)
        amount_received += len(data)
        print('Received "%s"' % data,file=sys.stderr)
        
finally:
    print('Closing socket',file=sys.stderr)
    sock.close()