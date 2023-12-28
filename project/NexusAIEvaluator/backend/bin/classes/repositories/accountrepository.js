const { dbConnect } = require("../database/database");
const FILEUtil = require("../utils/FileUtil");

module.exports = {
  async getHashedPassword(usrname) {
    return await dbConnect()
      .then((client) => {
        let args = [];
        args.push(usrname);
        return client
          .query(FILEUtil.getFiledataAsString(__dirname+"/sql/getPasswordHash.sql"), args)
          .then((hashed_psw) => {
            if (hashed_psw.rows.length > 0) {
              return hashed_psw.rows[0].hashed_psw;
            }
            return Promise.reject(false);
          })
          .catch((err) => {
            return Promise.reject(false);
          });
      })
      .catch((err) => {
        return Promise.reject(err);
      });
  }
};