pub trait Draw {
    fn draw(&self);
}

pub struct Screen {
    pub components: Vec <Box<dyn Draw>>,
}

impl Screen {
    pub fn run(&self) {
        for component in self.components.iter(){
            component.draw();
        }
    }
}

pub struct Button {
    pub w: u32,
    pub h: u32,
    pub label: String,
}

impl Draw for Button{
    fn draw(&self){
        println!("Drawing button {}!", self.label);
    }
}

pub struct SelectBox {
    w: u32,
}

impl Draw for SelectBox {
    fn draw(&self) {
        println!("Draw selectbox {}!", self.w);
    }
}

fn main() {
    let screen = Screen {
        components: vec![
            Box::new(SelectBox {
                w: 75,
           }),
            Box::new(Button {
                w: 50,
                h: 10,
                label: String::from("OK"),
            }),
        ],
    };

    screen.run();
}

