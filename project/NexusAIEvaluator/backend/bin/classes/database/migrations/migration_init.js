var db = require("../database");
var appConf = require("../../appconfig");
var hashUtil = require("../../utils/hashutil");
const FILEUTIL = require("../../utils/FileUtil");

module.exports = {
    runMigration(){
        db.dbConnect().then(pool => {
            pool.on('connect', (client) => {
              client.query('SELECT NOW()').then(result =>{
                console.log("Database connection established!");
                console.log(result.rows[0].now);
                console.log("Preparing database...");
          
                try{
                  sql = FILEUTIL.getFiledataAsString("./database.sql");
                  client.query(sql).then(res=>{
                    // Creating the admin user if not existing
                    adminUser = FILEUTIL.getFiledataFromJsonToObjects("./bin/configs/default_admin_user.json");
                    const args = [];
                    args.push(adminUser.username);
                    // does the admin user exist?
                    client.query('SELECT username from users where username=$1',args).then(result=>{
                        if(result.rows.length == 0){
                          const appConfig = appConf.getConf();
                          let saltRounds = 8;
                          if(appConf != null){
                              saltRounds = Number(appConfig.psw_salt_length);
                          }
                            args.push(hashUtil.genHash(adminUser.password,saltRounds));
                            client.query('INSERT INTO USERS(username,hashed_psw) VALUES($1,$2)',args).then(u=>{
                                args.pop();
                                client.query('SELECT userid from users where username=$1',args).then(uid=>{
                                    if(uid.rows.length > 0){
                                        const userId = uid.rows[0].userid;
                                        console.log("user "+JSON.stringify(args)+" is added to the database!");
                                        client.query('SELECT roleid,rolename from roles').then(role=>{
                                            role.rows.forEach(rid => {
                                                const roleId = rid.roleid;
                                                const roleName = rid.rolename;
                                                let usrRoleArgs = [];
                                                usrRoleArgs.push(userId);
                                                usrRoleArgs.push(roleId);
                                                client.query('INSERT INTO user_roles(userid,roleid) VALUES($1,$2)',usrRoleArgs).then(res=>{
                                                    console.log("user "+JSON.stringify(args)+" has given the role "+roleName+".");
                                                }).catch(er=>console.error(er.message));
                                            });
                                        }).catch(er=>console.error(er.message));
                                    }
                                }).catch(er=>console.error(er.message));
                            }).catch(er=>console.error(er.message));
                        }
                    }).catch(e=>console.error(e.message));
                  }).catch(e=>console.error(e.message));
                }catch(err){
                  console.error(err.message);
                }
                console.log("Migration is completed!");
              });

              

            });
            
            pool.connect();
          }).catch(err=>{
              console.log(err.message);
          });
    }
}