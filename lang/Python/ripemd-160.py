import hashlib
h = hashlib.new('ripemd160')
h.update(b"Rosetta Code")
h.hexdigest()
'b3be159860842cebaa7174c8fff0aa9e50a5199f'

