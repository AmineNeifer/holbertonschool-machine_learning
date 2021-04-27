-- reates a trigger that resets the attribute valid_email only when the email has been changed.
CREATE TRIGGER email_check
	BEFORE UPDATE
	ON users
	FOR EACH ROW
	BEGIN
  		IF STRCMP(old.email, new.email) <> 0 THEN
			SET new.valid_email = 0;
		END IF;
	END;