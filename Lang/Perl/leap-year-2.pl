sub isleap { !($_[0] % 100) ? !($_[0] % 400) : !($_[0] % 4) }