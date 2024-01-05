var PG = require("pg");
var Promise = require("bluebird");
var fs = Promise.promisifyAll(require("fs"));

module.exports = {
  dbConnect() {
    return new Promise((res,rej) =>{
      try{
        const dbconf = JSON.parse(fs.readFileSync("./configs/dbconnect.json",
        "utf8"));  
        try{
          res( new PG.Pool({
          user: dbconf.username,
          host: dbconf.host,
          database: dbconf.database,
          password: dbconf.password,
          port: dbconf.port,
        }));
      }catch(err){
        console.error(err.message);
        return rej(err.message);
      }
      }catch(err){
        console.error("configs/dbconnect.json not found.");
      }
      return rej("configs/dbconnect.json not found.");
    });
  }};
