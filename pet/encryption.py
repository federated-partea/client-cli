from Crypto.Cipher import PKCS1_OAEP, AES
from Crypto.PublicKey import RSA
from Crypto.Random import get_random_bytes

from serialize import serialize, deserialize


class EncryptionObject:
    def __init__(self, enc_session_key, nonce, tag, ciphertext):
        self.enc_session_key = enc_session_key
        self.nonce = nonce
        self.tag = tag
        self.ciphertext = ciphertext


class Encryption:
    def __init__(self):
        self.private_key = RSA.generate(2048)
        self.public_key = self.private_key.publickey().export_key()
        self.public_keys = None

    def decrypt(self, enc_obj: EncryptionObject) -> any:
        # Decrypt the session key with the private RSA key
        cipher_rsa = PKCS1_OAEP.new(self.private_key)
        session_key = cipher_rsa.decrypt(enc_obj.enc_session_key)

        # Decrypt the data with the AES session key
        cipher_aes = AES.new(session_key, AES.MODE_EAX, enc_obj.nonce)
        decrypted_data = deserialize(cipher_aes.decrypt_and_verify(enc_obj.ciphertext, enc_obj.tag))

        return decrypted_data

    def decrypt_incoming(self, incoming_data: any):
        decrypted_data = []
        for data in incoming_data:
            decrypted_data.append([self.decrypt(data[0])])

        return decrypted_data


def encrypt(data: any, public_recepient_key):
    # Encrypt the session key with the public RSA key
    session_key = get_random_bytes(16)
    cipher_rsa = PKCS1_OAEP.new(RSA.import_key(public_recepient_key))
    enc_session_key = cipher_rsa.encrypt(session_key)

    # Encrypt the data with the AES session key
    cipher_aes = AES.new(session_key, AES.MODE_EAX)
    ciphertext, tag = cipher_aes.encrypt_and_digest(serialize(data))

    return EncryptionObject(enc_session_key, cipher_aes.nonce, tag, ciphertext)


def encrypt_outgoing(data: any, public_keys: {}) -> {}:
    encrypted_data = {}
    if len(public_keys.keys()) == 1:
        client_id = list(public_keys.keys())[0]
        encrypted_data[client_id] = encrypt(data=data, public_recepient_key=public_keys[client_id])
    else:
        i = 0
        for client_id in public_keys.keys():
            encrypted_data[client_id] = encrypt(data=data[i], public_recepient_key=public_keys[client_id])
            i += 1

    return encrypted_data
