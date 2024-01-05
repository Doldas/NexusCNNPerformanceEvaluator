const roleRepo = require("../classes/repositories/rolerepository");
const jwt = require("jsonwebtoken");
const fileUtil = require("../classes/utils/FileUtil");

module.exports = {
  async getRoles() {
    return await roleRepo.getRoles().then((roles)=>{
        console.log(roles);
        return roles;
      }).catch((err)=>{
        return Promise.reject(err);
      });
  },
  async hasRole(user_token, rolename){
    let result = jwt.verify(user_token, fileUtil.getFiledataFromJsonToObjects("configs/secret.json").jwtsecret, (err, authData) => {
      if (err)
        return false;
      else {
        return roleRepo.hasRole(authData.username, rolename);
      }
    });
    return result;
  }
};