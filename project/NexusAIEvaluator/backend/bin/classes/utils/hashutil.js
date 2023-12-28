const bcrypt = require("bcrypt");


module.exports = {
    genHash(psw,saltRounds=8){
        return bcrypt.hashSync(psw, saltRounds);
    },
    validateHash(hash,psw) {
        return bcrypt.compareSync(psw, hash);     
    }
};