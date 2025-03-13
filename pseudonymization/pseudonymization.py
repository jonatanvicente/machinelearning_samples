import pandas as pd
import uuid
import hashlib

data = pd.DataFrame({
    'name': ['Anna', 'John', 'Luigi'],
    'email': ['anna@example.com', 'john@example.com', 'luigi@example.com'],
    'location': ['New York', 'London', 'Barcelona']
})

id_pseudo = []

for n in range(len(data)):
    id_pseudo.append(str(uuid.uuid4()))

data['id_pseudo'] = id_pseudo
data.drop('name', axis=1, inplace=True)

def hash_data(data):
    return hashlib.sha256(data.encode()).hexdigest()

emails = []

for email in data['email']:
    h_email = hash_data(email)
    emails.append(h_email)
data['email'] = emails

my_tokens = {}

def tokenize(data):
    token = str(uuid.uuid4())
    my_tokens[token] = data
    return token

def recover_data(token):
    return my_tokens.get(token, "Token invalid")

original_data = "123-456-789"
token = tokenize(original_data)

print(f'Token generated: {token}')
print(f'Data recovered: {recover_data(token)}')

