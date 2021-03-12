import http.client

connection = http.client.HTTPSConnection('www.example.com',cert_file='myCert.PEM')
connection.request('GET','/index.html')
response = connection.getresponse()
data = response.read()
