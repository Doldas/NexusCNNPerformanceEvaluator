const { dbConnect } = require("../classes/database/database");
const hashUtil = require("../classes/utils/hashutil");
const accountRepo = require("../classes/repositories/accountrepository");
const roleCtrl = require("./rolecontroller");

module.exports = {
  async login(usrname, psw) {
    return await accountRepo.getHashedPassword(usrname,psw).then((psw_hashed)=>{
      return hashUtil.validateHash(psw_hashed, psw);
    }).catch((err)=>{
      return Promise.reject(err);
    });
  },
  async hasRole(usrname, rolename){
    return roleCtrl.hasRole(usrname,rolename);
  }
};
