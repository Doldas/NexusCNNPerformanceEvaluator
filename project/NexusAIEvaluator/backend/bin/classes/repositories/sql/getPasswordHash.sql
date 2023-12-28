-- $1 = username
SELECT hashed_psw FROM users where username=LOWER($1)