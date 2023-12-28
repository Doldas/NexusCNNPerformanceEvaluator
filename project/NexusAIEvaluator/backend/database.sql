CREATE TABLE IF NOT EXISTS users (
  userid SERIAL PRIMARY KEY,
  username varchar(45) UNIQUE NOT NULL,
  hashed_psw varchar(256) NOT NULL,
  is_enabled boolean DEFAULT true
);

CREATE TABLE IF NOT EXISTS roles (
  roleid SERIAL PRIMARY KEY,
  rolename varchar(45) NOT NULL
);

CREATE TABLE IF NOT EXISTS user_roles (
  userid SERIAL,
  roleid SERIAL,
  PRIMARY KEY(userid,roleid),
    FOREIGN KEY(userid) 
        REFERENCES users(userid)
        ON DELETE CASCADE,
    FOREIGN KEY(roleid) 
        REFERENCES roles(roleid)
        ON DELETE CASCADE
);

INSERT INTO roles
    (rolename)
SELECT 'Admin'
WHERE
    NOT EXISTS (
        SELECT rolename FROM roles WHERE rolename = 'Admin'
    );

INSERT INTO roles
    (rolename)
SELECT 'Default'
WHERE
    NOT EXISTS (
        SELECT rolename FROM roles WHERE rolename = 'Default'
    );