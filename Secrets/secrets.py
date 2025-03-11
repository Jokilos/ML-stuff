def read_secret(name):
    with open('/home/nojak/Documents/' + name, 'r') as file:
        secret = file.read()
    
    return secret 