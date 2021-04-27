DROP TRIGGER IF EXISTS trigger_add;
CREATE TRIGGER trigger_add
AFTER INSERT ON orders
FOR EACH ROW
UPDATE items SET quantity = quantity - NEW.number
WHERE items.name=NEW.item_name;