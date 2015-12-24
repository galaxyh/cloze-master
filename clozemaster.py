from mprpc import RPCClient

client = RPCClient('127.0.0.1', 1979)
print client.call('doesnt_match', ['apple', 'boat', 'banana'])
print client.call('similarity', 'man', 'woman')
print client.call('most_similar', ['woman', 'king'], ['male'], 10, None)
print client.call('vector', 'fish')
