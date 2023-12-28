const { dbConnect } = require("../database/database");
const FILEUtil = require("../utils/FileUtil");

module.exports = {
  async getRoles(){
    return await dbConnect().then((client) =>{
      client.query(FILEUtil.getFiledataAsString(__dirname+"/sql/getRoles.sql")).then(
        (roles)=>{
            let roleObjects = [];
            roles.rows.forEach((roleEntity) =>{
              roleObjects.push({rolename:roleEntity.rolename});
            });
            return Promise.resolve(roleObjects);
        }
      ).catch((err)=>{
      return Promise.resolve([]);
    });

    }).catch((err)=>{
      return Promise.resolve([]);
    });
  },
  async hasRole(usrname,rolename) {
    return await dbConnect()
      .then((client) => {
        let args = [];
        args.push(usrname);
        args.push(rolename);
        return client
          .query(FILEUtil.getFiledataAsString(__dirname+"/sql/getRoleIdByUsername.sql"), args)
          .then((roleid) => {
            if(roleid.rows.length > 0){
                return Promise.resolve(true);
            }
            return Promise.resolve(false);
          })
          .catch((err) => {
            return Promise.resolve(false);
          });
      })
      .catch((err) => {
        return Promise.resolve(false);
      });
  }
};