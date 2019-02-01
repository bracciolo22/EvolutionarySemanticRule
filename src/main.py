from cheroot.wsgi import Server as WSGIServer
import logging
from pathlib import Path
import sys


if __name__ == "__main__":
	logging.basicConfig(format='%(asctime)s %(levelname)s: %(message)s',
						level=logging.INFO)

	# Run server
	from seedFlask import app
	
	port = 5000
	ip = '0.0.0.0'
	print("to connect to the server, with your browser, go to ", "localhost" if ip == '0.0.0.0' else ip, ":", port, sep='')

	server = WSGIServer(
		bind_addr=(ip, port),
		wsgi_app=app,
		request_queue_size=500,
	)

	try:
		server.start()
	except KeyboardInterrupt:
		pass
	finally:
		server.stop()
