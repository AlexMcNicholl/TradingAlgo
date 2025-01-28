from ib_insync import IB

def test_ibkr_connection(host='127.0.0.1', port=7497, client_id=1):
    """
    Test connection to the IBKR API.
    :param host: Host IP address (default is '127.0.0.1')
    :param port: Port number (default is 7497 for TWS, 4002 for IB Gateway)
    :param client_id: Client ID for the connection (default is 1)
    """
    ib = IB()
    try:
        ib.connect(host, port, clientId=client_id)
        if ib.isConnected():
            print(f"Connection successful to IBKR API (Host: {host}, Port: {port}, Client ID: {client_id}).")
            print("Connection Statistics:")
            print(f"  Start Time: {ib.client.connectionStats().startTime}")
            print(f"  Duration: {ib.client.connectionStats().duration} seconds")
            print(f"  Bytes Received: {ib.client.connectionStats().numBytesRecv}")
            print(f"  Bytes Sent: {ib.client.connectionStats().numBytesSent}")
            print(f"  Messages Received: {ib.client.connectionStats().numMsgRecv}")
            print(f"  Messages Sent: {ib.client.connectionStats().numMsgSent}")
        else:
            print("Connected but failed to retrieve connection details.")
    except Exception as e:
        print(f"Failed to connect to IBKR API. Error: {e}")
    finally:
        ib.disconnect()

if __name__ == "__main__":
    test_ibkr_connection()
