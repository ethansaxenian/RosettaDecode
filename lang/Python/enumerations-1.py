from enum import Enum
Contact = Enum('Contact', 'FIRST_NAME, LAST_NAME, PHONE')
Contact.__members__
mappingproxy(OrderedDict([('FIRST_NAME', Contact.FIRST_NAME), ('LAST_NAME', Contact.LAST_NAME), ('PHONE', Contact.PHONE)]))

# Explicit
class Contact2(Enum):
	FIRST_NAME = 1
	LAST_NAME = 2
	PHONE = 3

	
Contact2.__members__
mappingproxy(OrderedDict([('FIRST_NAME', Contact2.FIRST_NAME), ('LAST_NAME', Contact2.LAST_NAME), ('PHONE', Contact2.PHONE)]))

