from mprpc import RPCClient

client = RPCClient('127.0.0.1', 1979)
print client.call('doesnt_match', 'apple boat banana orange')
print client.call('vector', 'fish')
print client.call('most_similar', positive=['woman', 'king'], negative=['man'])
